#!/usr/bin/env python3
"""Fail-closed numeric and scope audit for the ForkAudit manuscript.

This script does not reinterpret GPU outputs.  It reconstructs the headline
quantities from every frozen raw shard, checks them against the aggregate and
derived artifacts, and verifies that the English manuscript contains the exact
bounded numbers, measurement windows, and disclaimers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/gpu-qwen35-vllm-paged-multifork-resident-20260814a"
SUMMARY = RAW / "multifork-resident-summary.json"
SHARDS = RAW / "resident-shards"
MANUSCRIPT = PAPER / "main.tex"
OUTPUT = PAPER / "generated/manuscript_evidence_audit.json"
ANONYMOUS = PAPER / "supplement_anonymous/raw_primary"
ANONYMOUS_SUMMARY = ANONYMOUS / "multifork-resident-summary.json"
ANONYMOUS_SHARDS = ANONYMOUS / "resident-shards"
ANONYMOUS_MANIFEST = PAPER / "supplement_anonymous/ANONYMOUS_MANIFEST.sha256"
ARTIFACT_METRICS = PAPER / "generated/artifact_metrics.json"
GDN_ORACLE_VALIDATION = (
    PAPER / "evidence/gdn_transition_oracle_preregistered_20260819d/validation_report.json"
)
A4_ROOT = PAPER / "evidence/round6_a4_transformers_transfer_20260819b/results"
A4_AGGREGATE = A4_ROOT / "forkaudit-transformers-transfer-aggregate.json"
A4_ARTIFACT_LEDGER = A4_ROOT / "artifact-ledger.json"
RR2_DERIVED = PAPER / "evidence/round_04_rr2_package/derived/derived_summary_v2.json"
SERVING_PANEL = PAPER / "evidence/related_work_same_protocol/serving_panel_summary.json"
HYPIC_SUMMARY = (
    PAPER
    / "evidence/related_work_same_protocol/hypic-same-protocol-20260821c/summary.json"
)
HYPIC_INDEPENDENT = (
    PAPER
    / "evidence/related_work_same_protocol/hypic-same-protocol-20260821c/independent-summary.json"
)
HYPIC_STORE_ACCEPTANCE = (
    PAPER
    / "evidence/related_work_same_protocol/hypic-retained-state-r34-trial1892234/acceptance.json"
)
ASSURANCE_BOUNDARY = PAPER / "evidence/assurance_boundary_summary.json"
INTEGRATED_RESULTS = PAPER / "evidence/integrated_results.json"
EXPERIMENT_REGISTRY = PAPER / "evidence/experiment_registry.json"
CLAIM_EVIDENCE_MAP = PAPER / "evidence/claim_evidence_map.tsv"
METHOD_PROVENANCE = PAPER / "evidence/method_provenance.tsv"
R29_OBSERVER_ROOT = (
    PAPER / "evidence/r29_independent_observer/formal_run_20260825a"
)
R29_OBSERVER_PREREGISTRATION = (
    R29_OBSERVER_ROOT / "preregistration/preregistration.json"
)
R29_OBSERVER_SOURCE_LEDGER = R29_OBSERVER_ROOT / "preregistration/source-code.sha256"
R29_OBSERVER_AMENDMENT = (
    PAPER / "evidence/r29_independent_observer/preexecution-amendment-v2.json"
)
R29_OBSERVER_RESULT = (
    R29_OBSERVER_ROOT / "raw/independent-gdn-observer-result.json"
)
R29_OBSERVER_REPLAY = (
    R29_OBSERVER_ROOT / "replay/independent-gdn-observer-replay.json"
)
R29_OBSERVER_TERMINAL_LEDGER = (
    R29_OBSERVER_ROOT / "receipts/terminal-files.sha256"
)
R29_TWO_STREAM_ROOT = PAPER / "evidence/r29_true_concurrency/formal_run_20260825b"
R29_TWO_STREAM_DESIGN = (
    PAPER / "evidence/r29_true_concurrency/design_preregistration.json"
)
R29_TWO_STREAM_AMENDMENT = (
    PAPER / "evidence/r29_true_concurrency/pre-second-execution-amendment-v3.json"
)
R29_TWO_STREAM_SOURCE_LEDGER = (
    PAPER / "evidence/r29_true_concurrency/source-code-v3.sha256"
)
R29_TWO_STREAM_RESULT = R29_TWO_STREAM_ROOT / "raw/formal-result.json"
R29_TWO_STREAM_REPLAY = R29_TWO_STREAM_ROOT / "replay/independent-replay.json"
R29_TWO_STREAM_TERMINAL_LEDGER = (
    R29_TWO_STREAM_ROOT / "receipts/raw-and-replay.sha256"
)
R29_LOCAL_OVERHEAD_ROOT = PAPER / "evidence/r29_overhead"
R29_LOCAL_OVERHEAD_PREREGISTRATION = R29_LOCAL_OVERHEAD_ROOT / "preregistration.json"
R29_LOCAL_OVERHEAD_RESULT = R29_LOCAL_OVERHEAD_ROOT / "local_replay_result.json"
R29_LOCAL_OVERHEAD_VALIDATION = R29_LOCAL_OVERHEAD_ROOT / "validation_report.json"
R29_LOCAL_OVERHEAD_TERMINAL_LEDGER = (
    R29_LOCAL_OVERHEAD_ROOT / "terminal-files.sha256"
)
R29_LIVE_OVERHEAD_ROOT = PAPER / "evidence/r29_live_overhead"
R29_LIVE_OVERHEAD_RUN = R29_LIVE_OVERHEAD_ROOT / "formal_run_20260825b"
R29_LIVE_OVERHEAD_PREREGISTRATION = R29_LIVE_OVERHEAD_ROOT / "preregistration.json"
R29_LIVE_OVERHEAD_PRESECOND_AMENDMENT = (
    R29_LIVE_OVERHEAD_ROOT / "pre-second-execution-amendment-v2.json"
)
R29_LIVE_OVERHEAD_REPLAY_AMENDMENT = (
    R29_LIVE_OVERHEAD_ROOT / "postexecution-replay-only-amendment-v2.json"
)
R29_LIVE_OVERHEAD_SOURCE_LEDGER = (
    R29_LIVE_OVERHEAD_ROOT / "source-code-replay-v2.sha256"
)
R29_LIVE_OVERHEAD_RESULT = R29_LIVE_OVERHEAD_RUN / "raw/formal-result.json"
R29_LIVE_OVERHEAD_SEMANTIC_SIDECAR = (
    R29_LIVE_OVERHEAD_RUN / "raw/audit/semantic-logits.fp32.bin"
)
R29_LIVE_OVERHEAD_REPLAY = (
    R29_LIVE_OVERHEAD_RUN / "replay/independent-replay-v2.json"
)
R29_LIVE_OVERHEAD_TERMINAL_LEDGER = (
    R29_LIVE_OVERHEAD_RUN / "receipts/raw-and-replay-v2.sha256"
)
R29_LIVE_OVERHEAD_STAGE03 = (
    R29_LIVE_OVERHEAD_RUN / "stages/03-independent-replay-v2-complete"
)
R29_LIVE_OVERHEAD_COMPLETED = R29_LIVE_OVERHEAD_RUN / "stages/COMPLETED-v2"
R29_HELDOUT_ROOT = PAPER / "evidence/r29_heldout_faults"
R29_HELDOUT_SUITE = R29_HELDOUT_ROOT / "preregistration/heldout-fault-suite.json"
R29_HELDOUT_AUTHOR_FREEZE = (
    R29_HELDOUT_ROOT / "preregistration/author-freeze-receipt.json"
)
R29_HELDOUT_AUTHOR_LEDGER = (
    R29_HELDOUT_ROOT / "preregistration/fault-author-code.sha256"
)
R29_HELDOUT_EXECUTION_INPUT_V3 = (
    R29_HELDOUT_ROOT / "cross_execution/execution-input-v3.json"
)
R29_HELDOUT_AMENDMENT_V3 = (
    R29_HELDOUT_ROOT / "cross_execution/pre-third-execution-amendment-v3.json"
)
R29_HELDOUT_EXECUTOR_LEDGER_V3 = (
    R29_HELDOUT_ROOT / "cross_execution/executor-source-v3.sha256"
)
R29_HELDOUT_READINESS_V3 = (
    R29_HELDOUT_ROOT / "cross_execution/readiness_report-v3.json"
)
R29_HELDOUT_FREEZE_V3 = (
    R29_HELDOUT_ROOT / "cross_execution/preexecution-freeze-v3.sha256"
)
R30_ROOT = PAPER / "evidence/r30_expanded_oracle_sweep"
R30_PREREGISTRATION = R30_ROOT / "preregistration.json"
R30_INPUT_MANIFEST = R30_ROOT / "input-manifest.json"
R30_PREEXECUTION_PIN = R30_ROOT / "preexecution-pin.json"
R30_SOURCE_LEDGER = R30_ROOT / "preexecution-source-ledger.sha256"
R30_CAPTURE_MANIFEST = R30_ROOT / "capture-manifest.json"
R30_RAW_CAPTURE_MANIFEST = R30_ROOT / "raw/capture-manifest.json"
R30_ORACLE_RESULT = R30_ROOT / "oracle-result.json"
R30_VALIDATION_REPORT = R30_ROOT / "validation_report.json"
R30_RAW_ARTIFACTS_LEDGER = R30_ROOT / "raw-artifacts.sha256"
R30_TERMINAL_PRODUCTS_LEDGER = R30_ROOT / "terminal-products.sha256"
R30_REFERENCE = R30_ROOT / "r30_expanded_oracle_reference.py"
R33_CAPTURE_ROOT = PAPER / "evidence/r33_independent_capture"
R33_CAPTURE_PREREGISTRATION = R33_CAPTURE_ROOT / "preregistration.json"
R33_CAPTURE_AMENDMENT = R33_CAPTURE_ROOT / "preexecution-amendment-v2.json"
R33_CAPTURE_SOURCE_LEDGER = R33_CAPTURE_ROOT / "source-code.sha256"
R33_CAPTURE_EXECUTION_PACKAGE = (
    R33_CAPTURE_ROOT / "r33_independent_capture_execution_v2.tar.gz"
)
R33_CAPTURE_FORMAL_ROOT = R33_CAPTURE_ROOT / "formal_h20"
R33_CAPTURE_RESULT_ARCHIVE = (
    R33_CAPTURE_FORMAL_ROOT / "r33-independent-capture-result.tar.gz"
)
R33_CAPTURE_RESULT_ROOT = R33_CAPTURE_FORMAL_ROOT / "result"
R33_CAPTURE_RESULT_PREREGISTRATION = (
    R33_CAPTURE_RESULT_ROOT / "preregistration/preregistration.json"
)
R33_CAPTURE_RESULT_SOURCE_LEDGER = (
    R33_CAPTURE_RESULT_ROOT / "preregistration/source-code.sha256"
)
R33_CAPTURE_RAW = (
    R33_CAPTURE_RESULT_ROOT / "raw/out-of-process-gdn-capture.json"
)
R33_CAPTURE_REPLAY = (
    R33_CAPTURE_RESULT_ROOT / "replay/out-of-process-gdn-replay.json"
)
R33_CAPTURE_TERMINAL_LEDGER = (
    R33_CAPTURE_RESULT_ROOT / "receipts/terminal-files.sha256"
)
R33_CAPTURE_COMPLETE_STAGE = R33_CAPTURE_RESULT_ROOT / "stages/06_complete"
R33_CAPTURE_ACCEPTANCE = R33_CAPTURE_FORMAL_ROOT / "independent_acceptance.json"
R33_HELDOUT_ROOT = PAPER / "evidence/r33_fresh_faults"
R33_HELDOUT_AUTHOR_ROOT = R33_HELDOUT_ROOT / "author_freeze"
R33_HELDOUT_FAULTS = R33_HELDOUT_AUTHOR_ROOT / "FAULTS.json"
R33_HELDOUT_AUTHOR_PROTOCOL = R33_HELDOUT_AUTHOR_ROOT / "PROTOCOL.md"
R33_HELDOUT_AUTHOR_MANIFEST = R33_HELDOUT_AUTHOR_ROOT / "MANIFEST.sha256"
R33_HELDOUT_DESIGNER_PDF = (
    R33_HELDOUT_AUTHOR_ROOT / "designer_input/round32_input.pdf"
)
R33_HELDOUT_DESIGNER_PDF_SHA = (
    R33_HELDOUT_AUTHOR_ROOT / "designer_input/round32_input.pdf.sha256"
)
R33_HELDOUT_EXECUTOR_ROOT = R33_HELDOUT_ROOT / "executor_attempt_b"
R33_HELDOUT_AMENDMENT = R33_HELDOUT_EXECUTOR_ROOT / "AMENDMENT.md"
R33_HELDOUT_FORMAL_PROTOCOL = R33_HELDOUT_EXECUTOR_ROOT / "formal-protocol.json"
R33_HELDOUT_PACKAGE_LEDGER = R33_HELDOUT_EXECUTOR_ROOT / "PACKAGE.sha256"
R33_HELDOUT_EXECUTION_PACKAGE = (
    R33_HELDOUT_EXECUTOR_ROOT / "r33-formal-launch-package-b.tar.gz"
)
R33_HELDOUT_EXECUTION_PACKAGE_SHA = (
    R33_HELDOUT_EXECUTOR_ROOT / "r33-formal-launch-package-b.tar.gz.sha256"
)
R33_HELDOUT_FORMAL_ROOT = R33_HELDOUT_ROOT / "formal_h20"
R33_HELDOUT_RESULT_VERIFICATION = (
    R33_HELDOUT_FORMAL_ROOT / "RESULT_VERIFICATION.json"
)
R33_HELDOUT_RESULT_ARCHIVE = (
    R33_HELDOUT_FORMAL_ROOT / "r33-fresh-faults-20260825b-result.tar.gz"
)
R33_HELDOUT_RESULT_ARCHIVE_SHA = (
    R33_HELDOUT_FORMAL_ROOT / "r33-fresh-faults-20260825b-result.tar.gz.sha256"
)
R33_HELDOUT_RUN_ROOT = (
    R33_HELDOUT_FORMAL_ROOT / "r33-fresh-faults-20260825b"
)
R33_HELDOUT_SUMMARY = R33_HELDOUT_RUN_ROOT / "summary.json"
R33_HELDOUT_TERMINAL_LEDGER = R33_HELDOUT_RUN_ROOT / "terminal-files.sha256"
R35_ROOT = PAPER / "evidence/r35_historical_alias_regression"
R35_DESIGN_DECISION = R35_ROOT / "design_decision.md"
R35_PREREGISTRATION = R35_ROOT / "preregistration.json"
R35_STATIC_EXECUTION_INPUT = R35_ROOT / "static-execution-input.json"
R35_SOURCE_LEDGER = R35_ROOT / "source.sha256"
R35_FREEZE_RECEIPT = R35_ROOT / "preexecution-freeze-receipt.json"
R35_RESOURCE_AMENDMENT = R35_ROOT / "preexecution-resource-amendment.json"
R35_EXECUTION_PACKAGE = R35_ROOT / "r35-execution-package.tar.gz"
R35_FORMAL_ROOT = R35_ROOT / "formal_h20"
R35_RESULT_VERIFICATION = R35_FORMAL_ROOT / "RESULT_VERIFICATION.json"
R35_RESULT_ARCHIVE = (
    R35_FORMAL_ROOT / "r35-historical-alias-20260826a-output.tar.gz"
)
R35_RUN_ROOT = R35_FORMAL_ROOT / "r35-historical-alias-20260826a"
R35_AGGREGATE = R35_RUN_ROOT / "aggregate.json"
R35_LAUNCH_COMPLETION = R35_RUN_ROOT / "launch-completion.json"
R35_VALIDATOR = PAPER / "scripts/validate_r35_historical_alias_evidence.py"
EXPECTED_HYPIC_STORE_ACCEPTANCE_SHA256 = "15dbee59e8f422a944cdcc2bd67c276b359b34327230569bd14f9afdb787cbec"
RENDERED_TABLE_INPUTS = (
    PAPER / "tables/h20_deployment_table.tex",
    PAPER / "tables/related_serving_table.tex",
    PAPER / "tables/related_work_reported_context.tex",
    PAPER / "tables/first_gate_localization_table.tex",
    PAPER / "tables/r33_fresh_heldout_table.tex",
    PAPER / "tables/r35_historical_alias_table.tex",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"{label} missing: {path}")
    observed = sha256(path)
    require(observed == expected, f"{label} SHA drift: {observed}")


def require_exact_fields(
    mapping: dict[str, object], expected: dict[str, object], label: str
) -> None:
    observed = {key: mapping.get(key) for key in expected}
    require(observed == expected, f"{label} drift: {observed}")


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def read_sha256_ledger(path: Path, label: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split("  ", 1)
        require(
            len(parts) == 2
            and re.fullmatch(r"[0-9a-f]{64}", parts[0]) is not None
            and bool(parts[1]),
            f"{label} malformed line {line_number}",
        )
        rows.append((parts[0], parts[1]))
    require(bool(rows), f"{label} is empty")
    require(
        len({name for _, name in rows}) == len(rows),
        f"{label} contains duplicate paths",
    )
    return rows


def verify_relative_sha256_ledger(
    ledger: Path,
    root: Path,
    label: str,
    *,
    excluded_relative_paths: set[str] | None = None,
) -> int:
    excluded = excluded_relative_paths or set()
    observed_paths: set[str] = set()
    for expected, raw_relative in read_sha256_ledger(ledger, label):
        relative = Path(raw_relative)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"{label} unsafe path: {raw_relative}",
        )
        normalized = relative.as_posix()
        require(normalized not in excluded, f"{label} unexpectedly lists {normalized}")
        require_sha256(root / relative, expected, f"{label} entry {normalized}")
        observed_paths.add(normalized)
    expected_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != ledger
        and path.relative_to(root).as_posix() not in excluded
    }
    require(
        observed_paths == expected_paths,
        f"{label} coverage drift: missing={sorted(expected_paths - observed_paths)}, "
        f"extra={sorted(observed_paths - expected_paths)}",
    )
    return len(observed_paths)


def verify_relocated_absolute_sha256_ledger(
    ledger: Path, root: Path, label: str
) -> int:
    observed_paths: set[str] = set()
    for expected, archived_absolute in read_sha256_ledger(ledger, label):
        archived_path = Path(archived_absolute)
        require(archived_path.is_absolute(), f"{label} path is not absolute")
        matching_indices = [
            index for index, part in enumerate(archived_path.parts)
            if part == root.name
        ]
        require(
            len(matching_indices) == 1,
            f"{label} output-root binding drift: {archived_absolute}",
        )
        relative = Path(*archived_path.parts[matching_indices[0] + 1 :])
        require(relative.parts, f"{label} lists output root rather than a file")
        normalized = relative.as_posix()
        require_sha256(root / relative, expected, f"{label} entry {normalized}")
        observed_paths.add(normalized)
    expected_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != ledger
    }
    require(
        observed_paths == expected_paths,
        f"{label} coverage drift: missing={sorted(expected_paths - observed_paths)}, "
        f"extra={sorted(observed_paths - expected_paths)}",
    )
    return len(observed_paths)


def load_tsv_index(
    path: Path, expected_header: list[str], key: str, label: str
) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames == expected_header, f"{label} header drift")
        rows = list(reader)
    require(
        all(None not in row and all(value is not None for value in row.values()) for row in rows),
        f"{label} malformed row",
    )
    values = [row[key] for row in rows]
    require(len(values) == len(set(values)), f"{label} duplicate {key}")
    return {row[key]: row for row in rows}


def indexed_experiment_registry() -> dict[str, dict[str, object]]:
    registry = json.loads(EXPERIMENT_REGISTRY.read_text())
    require(
        registry.get("schema_version") == "forkaudit-snapshot-experiment-registry-v2",
        "experiment-registry schema drift",
    )
    experiments = registry.get("experiments")
    require(isinstance(experiments, list), "experiment-registry experiments missing")
    evidence_ids = [row.get("evidence_id") for row in experiments]
    require(
        all(isinstance(value, str) and value for value in evidence_ids)
        and len(evidence_ids) == len(set(evidence_ids)),
        "experiment-registry evidence IDs invalid or duplicated",
    )
    return {str(row["evidence_id"]): row for row in experiments}


def run_json_replay(command: list[str], expected_path: Path, label: str) -> None:
    completed = subprocess.run(
        command,
        cwd=PAPER,
        check=False,
        text=True,
        capture_output=True,
    )
    require(
        completed.returncode == 0,
        f"{label} failed: stdout={completed.stdout!r}, stderr={completed.stderr!r}",
    )
    output_path = Path(command[command.index("--output") + 1])
    require(output_path.is_file(), f"{label} did not emit output")
    require(
        json.loads(output_path.read_text()) == json.loads(expected_path.read_text()),
        f"{label} output differs from archived replay",
    )


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def validate_r33_capture_evidence(
    integrated: dict[str, object],
    experiments: dict[str, dict[str, object]],
    claims: dict[str, dict[str, str]],
    methods: dict[str, dict[str, str]],
) -> dict[str, object]:
    evidence_id = "E-R33-OUT-OF-PROCESS-GDN-CAPTURE-A"
    expected_sha256 = {
        R33_CAPTURE_PREREGISTRATION: "67c001a04e61006967befe1ff1e26018b58bc02630b8e48dd40f16067a44ea65",
        R33_CAPTURE_AMENDMENT: "b1ec6b9b7d0c38e2312c827084deffe9dfe1e1317391da57cef000e05ca70b78",
        R33_CAPTURE_SOURCE_LEDGER: "08dee9b0e7f92edd65472618ebc3e9c2a38108296930255ffb62248bbe853319",
        R33_CAPTURE_EXECUTION_PACKAGE: "12050c65a6720e1c8b502ef1c470396c2da01765cd6bd05cb458009ebaea4ffd",
        R33_CAPTURE_RESULT_ARCHIVE: "19ed96509734ac49a36b2b0a0864567f1c35944a6d4ac348c95867c6a18d392b",
        R33_CAPTURE_RESULT_PREREGISTRATION: "67c001a04e61006967befe1ff1e26018b58bc02630b8e48dd40f16067a44ea65",
        R33_CAPTURE_RESULT_SOURCE_LEDGER: "08dee9b0e7f92edd65472618ebc3e9c2a38108296930255ffb62248bbe853319",
        R33_CAPTURE_RAW: "50d39cfcea072fb770da539d90abeddcd8a40802b88f4f95315001333c09e974",
        R33_CAPTURE_REPLAY: "dfda58f7596643b6a7366f217123aba5f51a29b0e1e93408419f73176eda8180",
        R33_CAPTURE_TERMINAL_LEDGER: "0261aa174435b6e1affdbd2fc116e5b30b38c8ca4d1bf6b57ba2f6a6a10961ae",
        R33_CAPTURE_COMPLETE_STAGE: "496a181aa41197c1cde6b4f3b8977c0f63bba14f232223b6eab8bc741f6f3c59",
        R33_CAPTURE_ACCEPTANCE: "d730793e9cf57fabeccdc5d0dba16ef7ba2da8b64c3e74c109bdb4da5134d1b0",
    }
    for path, expected in expected_sha256.items():
        require_sha256(path, expected, f"R33 capture artifact {path.relative_to(PAPER)}")

    source_bindings = {
        "r33_ipc_capture_protocol.py": "e651fa7500358ba8c3165fcf1ed018df995f8b8d613d046fe5e08859efe3ffde",
        "r33_independent_capture_worker.py": "f9418d1eb72bedde95cfa9f1e1b3b2823263f885fcdc86950c0d480ee9db0ff8",
        "r33_out_of_process_capture.py": "d3b8ca26a6bfed374ab0078de7161003860d13d94f567f81c463c3bafc93cd02",
        "r33_replay_independent_capture.py": "20a90bcc13d4604e2ce59590926548e8906f219a2537087e61265fc1eaf63e09",
        "r33_run_h20_independent_capture.py": "076757e691aecabbae5c0c5226d481a0c3993006e9f1b27f767ec4667e171640",
        "r33_run_local_capture_gate.py": "d10121b0473c841a3691679c05baa0ca5f113df2ec1f022652c5025b3ecef8dd",
        "r33_test_independent_capture.py": "9164b3338c769a942736feb3937eb194bc1ed34fdf1fe0e0b2430be042d35ae8",
        "r33_launch_h20_independent_capture_1gpu.sh": "bfa135ea087205e50f8dd1713dfe6dba59918858070d24e714b8498574c2837a",
    }
    source_ledger_rows = read_sha256_ledger(
        R33_CAPTURE_SOURCE_LEDGER, "R33 capture source ledger"
    )
    require(
        {name: digest for digest, name in source_ledger_rows} == source_bindings,
        "R33 capture source ledger content drift",
    )
    for name, expected in source_bindings.items():
        require_sha256(PAPER / "scripts" / name, expected, f"R33 capture source {name}")
    terminal_count = verify_relative_sha256_ledger(
        R33_CAPTURE_TERMINAL_LEDGER,
        R33_CAPTURE_RESULT_ROOT,
        "R33 capture terminal ledger",
        excluded_relative_paths={"stages/06_complete"},
    )
    require(terminal_count == 17, "R33 capture terminal ledger count drift")

    preregistration = json.loads(R33_CAPTURE_PREREGISTRATION.read_text())
    require_exact_fields(
        preregistration,
        {
            "schema_version": "forkaudit-r33-out-of-process-preregistration-v1",
            "status": "v2_frozen_before_h20_execution_and_before_h20_candidate_outputs",
            "experiment_id": "R33-OUT-OF-PROCESS-GDN-CAPTURE-20260825A",
            "preexecution_amendment_raw_sha256": expected_sha256[R33_CAPTURE_AMENDMENT],
        },
        "R33 capture preregistration identity",
    )
    require_exact_fields(
        preregistration.get("model", {}),
        {
            "id": "Qwen/Qwen3.5-35B-A3B",
            "revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
            "dtype": "bfloat16",
            "hardware": "one isolated NVIDIA H20-3e selected from an eight-H20 node",
        },
        "R33 capture fixed stack",
    )
    design = preregistration.get("design", {})
    require_exact_fields(
        design,
        {
            "resident_count": 2,
            "kv_policy_fixed": "vllm-q16-shared-document-reuse",
            "gdn_policy_cells": ["shared-base", "materialized"],
            "rows_per_phase": 180,
            "unordered_pair_relations_per_phase": 16_110,
            "total_row_observations": 1_080,
            "total_pair_relations": 96_660,
            "observer_processes": 2,
            "expected_transport": "torch-cuda-ipc-reduction",
        },
        "R33 capture preregistered design",
    )
    expected_capture_plan = [
        {
            "capture_id": "c-2d8d91660bc7",
            "phase": "setup_pre_transition",
            "completed_request_indices": [],
        },
        {
            "capture_id": "c-7b91ee24a5d3",
            "phase": "post_transition",
            "completed_request_indices": [0],
        },
        {
            "capture_id": "c-f109e345a0c8",
            "phase": "post_generation",
            "completed_request_indices": [0, 1],
        },
    ]
    require(design.get("capture_plan") == expected_capture_plan, "R33 capture plan drift")
    require_exact_fields(
        preregistration.get("live_wire_contract", {}),
        {
            "top_level_fields": ["capture_id", "schema_version", "slot_tensors"],
            "slot_fields": ["slot_id", "tensor"],
            "explicitly_absent": [
                "phase",
                "policy",
                "completed_request_indices",
                "expected relations",
                "candidate rows",
                "candidate passed fields",
                "candidate verdicts",
            ],
        },
        "R33 capture live-wire contract",
    )
    independence = preregistration.get("independence_boundary", {})
    require_exact_fields(
        independence,
        {
            "same_process": False,
            "worker_candidate_imports": False,
            "worker_model_runtime_imports": False,
            "worker_receives_candidate_rows": False,
            "worker_receives_candidate_verdicts": False,
            "worker_receives_phase_or_policy_labels_live": False,
            "imported_views_pinned_against_receiver_aba": True,
            "raw_addresses_serialized": False,
        },
        "R33 capture independence boundary",
    )
    require_exact_fields(
        preregistration.get("reporting_rule", {}),
        {
            "any_descriptor_relation_or_lifecycle_mismatch_must_be_preserved": True,
            "candidate_passed_fields_are_authoritative": False,
            "claim_only_after_frozen_protocol_bound_cpu_replay": True,
        },
        "R33 capture reporting rule",
    )

    raw = json.loads(R33_CAPTURE_RAW.read_text())
    require_exact_fields(
        raw,
        {
            "schema_version": "forkaudit-r33-out-of-process-result-v1",
            "status": "completed_pending_independent_replay",
            "claim_authorized": False,
            "preregistration_sha256": expected_sha256[R33_CAPTURE_PREREGISTRATION],
            "source_ledger_raw_sha256": expected_sha256[R33_CAPTURE_SOURCE_LEDGER],
        },
        "R33 capture immutable raw boundary",
    )
    require(
        sha256_json(raw)
        == "c0d10ae4bb961ab194bf5dcd8beedd80b07a2cd629823a080de5497fac5497f6",
        "R33 capture canonical raw SHA drift",
    )
    require(raw.get("hardware", {}).get("name") == "NVIDIA H20-3e", "R33 capture hardware drift")
    require_exact_fields(
        raw.get("independence_boundary", {}),
        {
            "same_process": False,
            "worker_candidate_imports": False,
            "worker_model_runtime_imports": False,
            "worker_receives_candidate_rows": False,
            "worker_receives_candidate_verdicts": False,
            "worker_receives_phase_or_policy_labels_live": False,
            "imported_views_pinned_against_receiver_aba": True,
            "raw_addresses_serialized": False,
        },
        "R33 capture raw independence receipt",
    )
    cells = raw.get("cells")
    require(isinstance(cells, list) and len(cells) == 2, "R33 capture cell count drift")
    expected_manifest_sha256 = "e4ba057c78eb2537095f878d66aadede05a80cfe0ae64464e5f29c477b72a0da"
    expected_pids = {"shared-base": 2178, "materialized": 2384}
    expected_row_keys = {
        "byte_end_exclusive", "byte_start", "content_sha256", "device", "dtype",
        "layer_index", "owner_kind", "request_index", "shape", "slot_id",
        "state_family", "state_index", "storage_nbytes", "storage_offset",
        "storage_token", "stride", "tensor_nbytes", "view_token",
    }
    layer_indices = design.get("linear_layer_indices")
    require(isinstance(layer_indices, list) and len(layer_indices) == 30, "R33 GDN layer list drift")
    expected_coordinates = {
        (layer, family, owner, request_index, 0)
        for layer in layer_indices
        for family in ("conv", "recurrent")
        for owner, request_index in (
            ("persistent", None), ("request", 0), ("request", 1)
        )
    }
    first_manifest: dict[str, object] | None = None
    observer_pids: list[int] = []
    total_rows = 0
    total_relations = 0
    total_model_steps = 0
    total_kernel_receipts = 0
    for cell in cells:
        policy = cell.get("policy")
        require(policy in expected_pids, f"R33 capture unexpected policy: {policy}")
        require(cell.get("capture_plan") == expected_capture_plan, f"R33 {policy} capture-plan drift")
        manifest = cell.get("slot_manifest")
        require(isinstance(manifest, dict), f"R33 {policy} manifest missing")
        require(
            manifest.get("schema_version") == "forkaudit-r33-ipc-slot-manifest-v1"
            and manifest.get("manifest_sha256") == expected_manifest_sha256
            and sha256_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})
            == expected_manifest_sha256,
            f"R33 {policy} manifest digest drift",
        )
        if first_manifest is None:
            first_manifest = manifest
        else:
            require(manifest == first_manifest, "R33 cell manifests differ")
        manifest_slots = manifest.get("slots")
        require(isinstance(manifest_slots, list) and len(manifest_slots) == 180, f"R33 {policy} slot count drift")
        manifest_by_id = {row.get("slot_id"): row for row in manifest_slots}
        require(len(manifest_by_id) == 180 and None not in manifest_by_id, f"R33 {policy} slot IDs invalid")
        manifest_coordinates = {
            (
                row.get("layer_index"), row.get("state_family"), row.get("owner_kind"),
                row.get("request_index"), row.get("state_index"),
            )
            for row in manifest_slots
        }
        require(manifest_coordinates == expected_coordinates, f"R33 {policy} slot-coordinate grid drift")
        ready = cell.get("observer_ready_receipt", {})
        producer_pid = ready.get("producer_pid")
        observer_pid = ready.get("observer_pid")
        require_exact_fields(
            ready,
            {
                "schema_version": "forkaudit-r33-ipc-response-v1",
                "kind": "ready",
                "producer_pid": 1816,
                "observer_pid": expected_pids[policy],
                "process_separated": True,
                "candidate_modules_imported": False,
                "observer_generates_verdicts": False,
                "slot_manifest_sha256": expected_manifest_sha256,
            },
            f"R33 {policy} observer-ready receipt",
        )
        require(producer_pid != observer_pid, f"R33 {policy} PID separation failed")
        observer_pids.append(int(observer_pid))
        stop = cell.get("observer_stop_receipt", {})
        require_exact_fields(
            stop,
            {
                "schema_version": "forkaudit-r33-ipc-response-v1",
                "kind": "stopped",
                "observer_pid": expected_pids[policy],
                "capture_count": 3,
                "pinned_capture_count": 3,
            },
            f"R33 {policy} observer-stop receipt",
        )
        captures = cell.get("captures")
        require(isinstance(captures, list) and len(captures) == 3, f"R33 {policy} capture count drift")
        session_commitments: set[str] = set()
        for capture, planned in zip(captures, expected_capture_plan):
            require_exact_fields(
                capture,
                {
                    "schema_version": "forkaudit-r33-out-of-process-capture-v1",
                    "capture_id": planned["capture_id"],
                    "producer_pid": 1816,
                    "observer_pid": expected_pids[policy],
                    "process_separated": True,
                    "transport": "torch-cuda-ipc-reduction",
                    "live_request_fields_received": ["capture_id", "schema_version", "slot_tensors"],
                    "live_slot_fields_received": ["slot_id", "tensor"],
                    "judgment_fields_received": [],
                    "candidate_verdict_fields_received": False,
                    "receiver_derived_descriptors": True,
                    "receiver_derived_relations": True,
                    "raw_addresses_serialized": False,
                    "imported_views_pinned_against_receiver_aba": True,
                    "slot_manifest_sha256": expected_manifest_sha256,
                    "row_count": 180,
                    "relation_count": 16_110,
                },
                f"R33 {policy} capture {planned['capture_id']}",
            )
            rows = capture.get("rows")
            require(isinstance(rows, list) and len(rows) == 180, f"R33 {policy} capture rows drift")
            require(sha256_json(rows) == capture.get("rows_sha256"), f"R33 {policy} rows digest drift")
            require({row.get("slot_id") for row in rows} == set(manifest_by_id), f"R33 {policy} capture slot set drift")
            for row in rows:
                require(set(row) == expected_row_keys, f"R33 {policy} descriptor fields drift")
                slot = manifest_by_id[row["slot_id"]]
                for field in ("layer_index", "state_family", "owner_kind", "request_index", "state_index"):
                    require(row[field] == slot[field], f"R33 {policy} slot binding drift: {field}")
                geometry = design["allowed_descriptor_geometry_by_family"][row["state_family"]]
                require(row["shape"] == geometry["shape"], f"R33 {policy} shape drift")
                require(row["stride"] in geometry["allowed_strides"], f"R33 {policy} stride drift")
                for field in ("storage_offset", "dtype", "storage_nbytes", "tensor_nbytes", "byte_start", "byte_end_exclusive"):
                    require(row[field] == geometry[field], f"R33 {policy} {field} drift")
                require(str(row["device"]).startswith(geometry["device_prefix"]), f"R33 {policy} device drift")
                for field in ("content_sha256", "storage_token", "view_token"):
                    require(re.fullmatch(r"[0-9a-f]{64}", row[field]) is not None, f"R33 {policy} {field} invalid")
            session_commitments.add(str(capture.get("observer_session_commitment_sha256")))
            total_rows += 180
            total_relations += 16_110
        require(len(session_commitments) == 1, f"R33 {policy} observer session drift")
        model_steps = cell.get("model_steps")
        kernel_receipts = cell.get("kernel_ledger_receipts")
        require(
            isinstance(model_steps, list) and len(model_steps) == 2
            and {row.get("request_index") for row in model_steps} == {0, 1},
            f"R33 {policy} model-step receipt drift",
        )
        require(
            isinstance(kernel_receipts, list) and len(kernel_receipts) == 2
            and all(row.get("verified") is True for row in kernel_receipts),
            f"R33 {policy} kernel-ledger receipt drift",
        )
        total_model_steps += len(model_steps)
        total_kernel_receipts += len(kernel_receipts)
    require(observer_pids == [2178, 2384], "R33 observer PID vector drift")
    require(total_rows == 1_080 and total_relations == 96_660, "R33 raw denominator drift")
    require(total_model_steps == 4 and total_kernel_receipts == 4, "R33 execution receipt count drift")

    replay = json.loads(R33_CAPTURE_REPLAY.read_text())
    require_exact_fields(
        replay,
        {
            "schema_version": "forkaudit-r33-out-of-process-replay-v1",
            "input_result_sha256": "c0d10ae4bb961ab194bf5dcd8beedd80b07a2cd629823a080de5497fac5497f6",
            "frozen_protocol_bound": True,
            "candidate_verdict_fields_authoritative": False,
            "cell_count": 2,
            "row_observations": 1_080,
            "relation_observations": 96_660,
            "all_observers_process_separated": True,
            "passed": True,
        },
        "R33 capture replay",
    )
    replay_cells = replay.get("cell_reports")
    require(isinstance(replay_cells, list) and len(replay_cells) == 2, "R33 replay cell count drift")
    require(
        sum(len(row.get("phase_reports", [])) for row in replay_cells) == 6
        and all(row.get("passed") is True for row in replay_cells)
        and all(
            phase.get("passed") is True
            for row in replay_cells
            for phase in row.get("phase_reports", [])
        ),
        "R33 phase/lifecycle verdict drift",
    )
    with tempfile.TemporaryDirectory(prefix="forkaudit-r33-capture-replay-") as temp_dir:
        replay_output = Path(temp_dir) / "replay.json"
        run_json_replay(
            [
                sys.executable,
                str(PAPER / "scripts/r33_replay_independent_capture.py"),
                "--input", str(R33_CAPTURE_RAW),
                "--expected-input-sha256", expected_sha256[R33_CAPTURE_RAW],
                "--preregistration", str(R33_CAPTURE_PREREGISTRATION),
                "--expected-preregistration-sha256", expected_sha256[R33_CAPTURE_PREREGISTRATION],
                "--output", str(replay_output),
            ],
            R33_CAPTURE_REPLAY,
            "R33 fresh CPU capture replay",
        )

    acceptance = json.loads(R33_CAPTURE_ACCEPTANCE.read_text())
    require_exact_fields(
        acceptance,
        {
            "schema_version": "forkaudit-r33-out-of-process-acceptance-v1",
            "status": "verified_bounded_pass",
            "evidence_id": evidence_id,
            "verdict": {
                "exact_bounded_claim": "resolved",
                "broader_trusted_capture_objection": "partially_resolved",
            },
        },
        "R33 capture independent acceptance identity",
    )
    require_exact_fields(
        acceptance.get("immutable_bindings", {}),
        {
            "preregistration_raw_sha256": expected_sha256[R33_CAPTURE_PREREGISTRATION],
            "source_ledger_raw_sha256": expected_sha256[R33_CAPTURE_SOURCE_LEDGER],
            "formal_result_file_sha256": expected_sha256[R33_CAPTURE_RAW],
            "formal_result_canonical_sha256": "c0d10ae4bb961ab194bf5dcd8beedd80b07a2cd629823a080de5497fac5497f6",
            "independent_replay_file_sha256": expected_sha256[R33_CAPTURE_REPLAY],
            "terminal_ledger_sha256": expected_sha256[R33_CAPTURE_TERMINAL_LEDGER],
            "terminal_complete_stage_sha256": expected_sha256[R33_CAPTURE_COMPLETE_STAGE],
            "result_archive_sha256": expected_sha256[R33_CAPTURE_RESULT_ARCHIVE],
            "execution_package_sha256": expected_sha256[R33_CAPTURE_EXECUTION_PACKAGE],
        },
        "R33 capture acceptance immutable bindings",
    )
    acceptance_verification = acceptance.get("verification", {})
    require_exact_fields(
        acceptance_verification,
        {
            "terminal_ledger_entries_verified": 17,
            "source_ledger_entries_verified": 8,
            "unit_tests_passed": 6,
            "policy_cells_passed": 2,
            "phase_verdicts_passed": 6,
            "lifecycle_verdicts_passed": 2,
            "row_observations": 1_080,
            "relation_observations": 96_660,
            "producer_pid": 1816,
            "observer_pids": [2178, 2384],
            "all_observers_process_separated": True,
            "transport": "torch-cuda-ipc-reduction",
            "live_request_fields": ["capture_id", "schema_version", "slot_tensors"],
            "live_slot_fields": ["slot_id", "tensor"],
            "judgment_fields_received": [],
            "candidate_verdict_fields_received": False,
            "receiver_derived_descriptors": True,
            "receiver_derived_relations": True,
            "raw_addresses_serialized": False,
            "imported_views_pinned_against_receiver_aba": True,
            "expected_manifest_sha256": expected_manifest_sha256,
            "both_cell_manifests_exactly_match_frozen_manifest": True,
            "all_six_capture_manifest_bindings_match": True,
            "all_six_slot_sets_complete_and_coordinate_bound": True,
            "archived_replay_exactly_matches_fresh_cpu_recomputation": True,
            "model_steps_completed": 4,
            "verified_kernel_ledger_receipts": 4,
        },
        "R33 capture acceptance verification",
    )
    authorization = acceptance.get("authorization", {})
    require_exact_fields(
        authorization,
        {
            "raw_status_preserved": "completed_pending_independent_replay",
            "raw_claim_authorized_preserved": False,
            "raw_or_replay_bytes_modified": False,
            "claim_authorized_at_acceptance_and_registry_layer": True,
        },
        "R33 capture authorization layering",
    )
    prohibited_expansions = [
        "malicious-producer resistance or producer-independent semantic-slot enumeration",
        "OS- or driver-level enumeration of CUDA allocations",
        "external ground truth or a trusted-computing-base-free monitor",
        "independent model execution or end-to-end semantic correctness",
        "independent KV ownership recapture",
        "attention-kernel, dispatcher, compiled-binary, or autotuning attestation",
        "continuous batching, production scheduling, throughput, latency, or capacity",
        "cross-model, cross-runtime, or cross-hardware generality",
        "coverage of the full 96-cell primary protocol",
        "exclusion of transient writes restored between paused captures",
    ]
    require(acceptance.get("prohibited_expansions") == prohibited_expansions, "R33 capture prohibited-expansion drift")

    r29_registry = experiments.get("E-R29-SOURCE-DISTINCT-GDN-OBSERVER", {})
    require_exact_fields(
        r29_registry,
        {
            "status": "verified_internal_superseded_by_r33_out_of_process_capture",
            "active_manuscript_support": False,
            "superseded_by_evidence_id": evidence_id,
        },
        "R29 capture registry supersession",
    )
    registry = experiments.get(evidence_id, {})
    require_exact_fields(
        registry,
        {
            "source_experiment_id": "R33-OUT-OF-PROCESS-GDN-CAPTURE-20260825A",
            "status": "verified_bounded_out_of_process_pytorch_cuda_ipc_gdn_recapture",
            "active_manuscript_support": True,
            "supersedes_evidence_id": "E-R29-SOURCE-DISTINCT-GDN-OBSERVER",
        },
        "R33 capture registry identity",
    )
    require_exact_fields(
        registry.get("source_binding", {}),
        {
            "preregistration_sha256": expected_sha256[R33_CAPTURE_PREREGISTRATION],
            "preexecution_amendment_sha256": expected_sha256[R33_CAPTURE_AMENDMENT],
            "source_ledger_sha256": expected_sha256[R33_CAPTURE_SOURCE_LEDGER],
            "formal_result_file_sha256": expected_sha256[R33_CAPTURE_RAW],
            "formal_result_canonical_sha256": "c0d10ae4bb961ab194bf5dcd8beedd80b07a2cd629823a080de5497fac5497f6",
            "independent_replay_sha256": expected_sha256[R33_CAPTURE_REPLAY],
            "terminal_ledger_sha256": expected_sha256[R33_CAPTURE_TERMINAL_LEDGER],
            "terminal_complete_stage_sha256": expected_sha256[R33_CAPTURE_COMPLETE_STAGE],
            "execution_package_sha256": expected_sha256[R33_CAPTURE_EXECUTION_PACKAGE],
            "result_archive_sha256": expected_sha256[R33_CAPTURE_RESULT_ARCHIVE],
            "independent_acceptance_sha256": expected_sha256[R33_CAPTURE_ACCEPTANCE],
        },
        "R33 capture registry source binding",
    )
    require_exact_fields(
        registry.get("validation", {}),
        {
            "scientific_run_valid": True,
            "fixed_model": "Qwen/Qwen3.5-35B-A3B",
            "fixed_model_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
            "hardware": "one NVIDIA H20-3e",
            "fresh_n2_policy_cells": 2,
            "phase_points": 6,
            "receiver_derived_row_observations": 1_080,
            "receiver_derived_pair_relations": 96_660,
            "phase_verdicts_passed": 6,
            "lifecycle_verdicts_passed": 2,
            "producer_pid": 1816,
            "observer_pids": [2178, 2384],
            "all_observers_process_separated": True,
            "transport": "torch-cuda-ipc-reduction",
            "live_judgment_fields_received": 0,
            "candidate_verdict_fields_received": False,
            "raw_addresses_serialized": False,
            "frozen_manifest_and_all_slot_coordinate_bindings_verified": True,
            "archived_replay_exactly_matches_fresh_cpu_recomputation": True,
            "model_steps_completed": 4,
            "verified_kernel_ledger_receipts": 4,
        },
        "R33 capture registry validation",
    )
    require_exact_fields(
        registry.get("authorization", {}),
        {
            "claim_authorized": True,
            "raw_status_preserved": "completed_pending_independent_replay",
            "raw_claim_authorized_preserved": False,
            "raw_or_replay_bytes_modified": False,
            "authorized_claim": authorization.get("authorized_claim"),
        },
        "R33 capture registry authorization",
    )
    require(
        len(registry.get("prohibited_expansions", [])) == 10,
        "R33 capture registry prohibited-expansion count drift",
    )

    integrated_capture = integrated.get("out_of_process_gdn_capture", {})
    require_exact_fields(
        integrated_capture,
        {
            "evidence_id": evidence_id,
            "status": "verified_bounded_out_of_process_pytorch_cuda_ipc_gdn_recapture",
            "active_manuscript_support": True,
            "preregistration_sha256": expected_sha256[R33_CAPTURE_PREREGISTRATION],
            "source_ledger_sha256": expected_sha256[R33_CAPTURE_SOURCE_LEDGER],
            "preexecution_amendment_sha256": expected_sha256[R33_CAPTURE_AMENDMENT],
            "formal_result_file_sha256": expected_sha256[R33_CAPTURE_RAW],
            "formal_result_canonical_sha256": "c0d10ae4bb961ab194bf5dcd8beedd80b07a2cd629823a080de5497fac5497f6",
            "independent_replay_sha256": expected_sha256[R33_CAPTURE_REPLAY],
            "terminal_ledger_sha256": expected_sha256[R33_CAPTURE_TERMINAL_LEDGER],
            "independent_acceptance_sha256": expected_sha256[R33_CAPTURE_ACCEPTANCE],
            "fixed_model": "Qwen/Qwen3.5-35B-A3B",
            "fixed_model_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
            "hardware": "one NVIDIA H20-3e",
            "fresh_n2_cells": 2,
            "phase_points": 6,
            "receiver_derived_rows": 1_080,
            "receiver_derived_pair_relations": 96_660,
            "phase_verdicts_passed": 6,
            "lifecycle_verdicts_passed": 2,
            "producer_pid": 1816,
            "observer_pids": [2178, 2384],
            "all_observers_process_separated": True,
            "transport": "torch-cuda-ipc-reduction",
            "live_judgment_fields_received": 0,
            "candidate_verdict_fields_received": False,
            "frozen_manifest_and_slot_bindings_verified": True,
            "fresh_cpu_replay_exactly_matches_archive": True,
            "claim_authorized": True,
            "raw_status_preserved": "completed_pending_independent_replay",
            "raw_claim_authorized_preserved": False,
            "authorized_claim": authorization.get("authorized_claim"),
        },
        "R33 integrated capture support",
    )
    integrated_boundary = str(integrated_capture.get("boundary", ""))
    for fragment in (
        "producer still enumerates and semantically binds frozen owner slots",
        "not malicious-producer resistance",
        "OS/driver allocation monitoring",
        "independent model or KV execution/capture",
        "cross-stack generality",
        "transient restored writes",
    ):
        require(fragment in integrated_boundary, f"R33 integrated capture boundary omits {fragment}")

    r29_claim = claims.get("C-R29-GDN-OBSERVER-01", {})
    require_exact_fields(
        r29_claim,
        {
            "status": "verified_internal_superseded_by_r33",
            "evidence_ids": "E-R29-SOURCE-DISTINCT-GDN-OBSERVER",
        },
        "R29 claim-map supersession",
    )
    require("Internal evidence registry only" in r29_claim.get("manuscript_locations", ""), "R29 observer remains visible support")
    capture_claim = claims.get("C-R33-OUT-OF-PROCESS-GDN-CAPTURE-01", {})
    require_exact_fields(
        capture_claim,
        {"status": "verified_bounded_active", "evidence_ids": evidence_id},
        "R33 capture claim-map identity",
    )
    capture_claim_text = " ".join(capture_claim.values())
    for fragment in (
        "1,080 storage-descriptor rows", "96,660 pair relations", "producer PID 1816",
        "observer PIDs 2178 and 2384", "zero judgment/verdict fields",
        "cross-model/runtime/hardware generality", "transient restored writes",
    ):
        require(fragment in capture_claim_text, f"R33 capture claim map omits {fragment}")
    rejected_capture = claims.get("C-UNSUPPORTED-R33-CAPTURE-EXPANSION", {})
    require_exact_fields(
        rejected_capture,
        {"status": "rejected", "evidence_ids": evidence_id},
        "R33 rejected capture expansion",
    )

    method_expectations = {
        "M-R33-IPC-WIRE-CONTRACT": "r33_ipc_capture_protocol.py",
        "M-R33-IPC-OBSERVER": "r33_independent_capture_worker.py",
        "M-R33-IPC-EXECUTION": "r33_out_of_process_capture.py",
        "M-R33-IPC-REPLAY-ACCEPTANCE": "r33_replay_independent_capture.py",
    }
    for method_id, source_fragment in method_expectations.items():
        row = methods.get(method_id, {})
        require(source_fragment in row.get("source_path", ""), f"{method_id} source provenance drift")
        require("Active narrow R33 support" in row.get("manuscript_locations", ""), f"{method_id} is not active narrow support")
    for method_id in (
        "M-R29-GDN-OBSERVER-CAPTURE",
        "M-R29-GDN-OBSERVER-EXECUTION",
        "M-R29-GDN-OBSERVER-REPLAY",
    ):
        require(
            "Internal evidence registry only" in methods.get(method_id, {}).get("manuscript_locations", ""),
            f"{method_id} remains visible capture support",
        )

    return {
        "evidence_id": evidence_id,
        "formal_result_sha256": expected_sha256[R33_CAPTURE_RAW],
        "formal_result_canonical_sha256": "c0d10ae4bb961ab194bf5dcd8beedd80b07a2cd629823a080de5497fac5497f6",
        "independent_replay_sha256": expected_sha256[R33_CAPTURE_REPLAY],
        "independent_acceptance_sha256": expected_sha256[R33_CAPTURE_ACCEPTANCE],
        "terminal_ledger_sha256": expected_sha256[R33_CAPTURE_TERMINAL_LEDGER],
        "terminal_files_verified": terminal_count,
        "fresh_cpu_replay_exactly_matches_archive": True,
        "raw_claim_authorized": False,
        "acceptance_and_registry_claim_authorized": True,
        "producer_pid": 1816,
        "observer_pids": observer_pids,
        "all_observers_process_separated": True,
        "receiver_derived_rows": total_rows,
        "receiver_derived_pair_relations": total_relations,
        "phase_verdicts_passed": 6,
        "lifecycle_verdicts_passed": 2,
        "prohibited_expansion_count": len(prohibited_expansions),
    }


def validate_r33_heldout_evidence(
    integrated: dict[str, object],
    experiments: dict[str, dict[str, object]],
    claims: dict[str, dict[str, str]],
    methods: dict[str, dict[str, str]],
) -> dict[str, object]:
    evidence_id = "E-R33-PDF-ONLY-FRESH-HELDOUT-FAULTS-B"
    run_id = "R33-FRESH-FAULTS-20260825B"
    expected_sha256 = {
        R33_HELDOUT_FAULTS: "b1f4d6c544c30fccc32370a03e170aee38596a370d02c0db4a6748c83cc34dff",
        R33_HELDOUT_AUTHOR_PROTOCOL: "b85995e180732588ac6ee09fc33181d9c276980795a7288a23feb4c94ad3925c",
        R33_HELDOUT_AUTHOR_MANIFEST: "132b794be970a61025dd1d63f26e9b0fbc978b00f2a9f94bf5864f5bb8f8c548",
        R33_HELDOUT_DESIGNER_PDF: "a34f319550300d603db259a69c5685112009b2d0a3d92aa3096a121624fb6db3",
        R33_HELDOUT_DESIGNER_PDF_SHA: "2b73866ed70f1ea5c78a29f0700a9d7adf8253c79438691d2d8c230fd67918bb",
        R33_HELDOUT_AMENDMENT: "24638a1043be7fda9af129255d398ca8c765c7f401219e11f8548b646c5dfda1",
        R33_HELDOUT_FORMAL_PROTOCOL: "7a5172e212e8cfb1541f7c8b901c72099a141ea9da55d2f7f14e4296cabb5ad4",
        R33_HELDOUT_PACKAGE_LEDGER: "7a9e1c6cfe1b51750d15108830591b1d3c2b66e3b882c0de3b1ae17cf3d1de8a",
        R33_HELDOUT_EXECUTION_PACKAGE: "69c860d084f03bf30f352aa04e454cfefeba9e4661b3d760652a679f7e7bf6cb",
        R33_HELDOUT_EXECUTION_PACKAGE_SHA: "2eedd91c159bd054ada6db846875f19f372b0018312271c06e3fa58d43585a3d",
        R33_HELDOUT_RESULT_VERIFICATION: "f92a872d8aece11b2f846b02f23632bf1b7ac6d8bee1982b99959d60c5bc8e89",
        R33_HELDOUT_RESULT_ARCHIVE: "0b8261a1c47dcf861379d2bb1629e84c804afb4f6719fdaa7e674a0dcef18441",
        R33_HELDOUT_RESULT_ARCHIVE_SHA: "c9a8c9b6d94fdda02afd78622832d1781825bf9d82acf4cab2e5ab9a5ebfb2f6",
        R33_HELDOUT_SUMMARY: "bdfb01bfb5211d70febd34aec0b7c96950b3955fd2913cb6aeb034b24b26a769",
        R33_HELDOUT_TERMINAL_LEDGER: "4038e0e163a8e592bfb0041566b452ff46b0a8709c3d33769827b44e3506a08d",
    }
    for path, expected in expected_sha256.items():
        require_sha256(path, expected, f"R33 held-out artifact {path.relative_to(PAPER)}")
    require(
        R33_HELDOUT_DESIGNER_PDF_SHA.read_text().strip()
        == f"{expected_sha256[R33_HELDOUT_DESIGNER_PDF]}  round32_input.pdf",
        "R33 held-out archived PDF digest receipt drift",
    )
    author_manifest_rows = read_sha256_ledger(
        R33_HELDOUT_AUTHOR_MANIFEST, "R33 held-out author manifest"
    )
    require(
        {name: digest for digest, name in author_manifest_rows}
        == {
            "FAULTS.json": expected_sha256[R33_HELDOUT_FAULTS],
            "PROTOCOL.md": expected_sha256[R33_HELDOUT_AUTHOR_PROTOCOL],
        },
        "R33 held-out author manifest content drift",
    )
    require(
        R33_HELDOUT_EXECUTION_PACKAGE_SHA.read_text().strip()
        == f"{expected_sha256[R33_HELDOUT_EXECUTION_PACKAGE]}  r33-formal-launch-package-b.tar.gz",
        "R33 held-out execution-package digest receipt drift",
    )
    require(
        R33_HELDOUT_RESULT_ARCHIVE_SHA.read_text().strip()
        == f"{expected_sha256[R33_HELDOUT_RESULT_ARCHIVE]}  r33-fresh-faults-20260825b-result.tar.gz",
        "R33 held-out result-archive digest receipt drift",
    )
    package_rows = read_sha256_ledger(
        R33_HELDOUT_PACKAGE_LEDGER, "R33 held-out Attempt-B package ledger"
    )
    require(len(package_rows) == 22, "R33 held-out package-ledger count drift")
    for expected, raw_relative in package_rows:
        relative = Path(raw_relative)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"R33 held-out package ledger unsafe path: {raw_relative}",
        )
        require_sha256(ROOT / relative, expected, f"R33 held-out package entry {raw_relative}")

    faults = json.loads(R33_HELDOUT_FAULTS.read_text())
    require_exact_fields(
        faults,
        {
            "schema_version": "r33-heldout-fault-freeze-v1",
            "freeze_status": "definitions_frozen_no_experiments_run",
            "author_role": "independent_held_out_fault_designer",
        },
        "R33 held-out author freeze identity",
    )
    source_audit = faults.get("source_audit", {})
    require_exact_fields(
        source_audit.get("sole_project_source_consulted", {}),
        {
            "sha256": expected_sha256[R33_HELDOUT_DESIGNER_PDF],
            "pages": 24,
            "access": [
                "complete pdftotext extraction",
                "visual inspection of rendered relevant pages",
            ],
        },
        "R33 held-out PDF-only source audit",
    )
    require(
        source_audit.get("not_consulted")
        == [
            "LaTeX or other paper sources",
            "implementation code or scripts",
            "pre-existing evidence directories or artifacts",
            "old fault-definition files",
            "review files",
            "state files",
        ],
        "R33 held-out excluded designer inputs drift",
    )
    require_exact_fields(
        faults.get("common_protocol", {}),
        {"no_gate_suppression": True, "no_experiments_run_by_author": True},
        "R33 held-out author protocol boundary",
    )
    fault_rows = faults.get("faults")
    require(isinstance(fault_rows, list) and len(fault_rows) == 5, "R33 held-out fault-count drift")
    expected_faults = [
        ("HF01_DELAYED_TAIL_DETACH", "TAIL_COPY_BEFORE_FIRST_APPEND_WRITE", "2ee27893a09cc9198f227422ec9fda1de1bebf97cc31b35fc1cfce67f773b8f2"),
        ("HF02_INACTIVE_DOCUMENT_LANE_SCRIBBLE", "PHYSICAL_DOCUMENT_PREFIX_IMMUTABLE", "20bbf518f3d2f66577db3e850400407658c8029975e03f6509a3e08f75d18970"),
        ("HF03_DUPLICATE_COMMITTED_DISPATCH", "ORDERED_CALL_CARDINALITY", "6e2b0b4cca4f8a3b72d26e2f13aa6a2a47c5791dd8df44c452fb99bd7d42f282"),
        ("HF04_EFFECTIVE_SCALE_DRIFT", "ATTENTION_EFFECTIVE_SCALE", "24c88a88ea2991d16f4e7e63c457fcf92d2a95650ef23232fcf2a1c24d7a64f7"),
        ("HF05_STALE_GDN_BINDING_TOKEN_AFTER_REBIND", "GDN_COMPLETED_BINDING_TOKEN_ADVANCE", "6dfbea24d869efeb4881155dfae1d710109a40017cb25bb3e80f621c266ec80a"),
    ]
    for row, (fault_id, _, expected_definition_sha256) in zip(fault_rows, expected_faults):
        require(row.get("id") == fault_id, f"R33 held-out fault order drift: {fault_id}")
        require(sha256_json(row) == expected_definition_sha256, f"R33 held-out fault definition drift: {fault_id}")

    formal_protocol = json.loads(R33_HELDOUT_FORMAL_PROTOCOL.read_text())
    require_exact_fields(
        formal_protocol,
        {
            "schema_version": "forkaudit-r33-execution-protocol-v1",
            "run_id": run_id,
            "mode": "formal_fresh_faults",
            "candidate_output_seen_when_frozen": False,
            "author_freeze_manifest_sha256": expected_sha256[R33_HELDOUT_AUTHOR_MANIFEST],
        },
        "R33 held-out Attempt-B formal protocol identity",
    )
    require_exact_fields(
        formal_protocol.get("claim_boundary", {}),
        {
            "autotuning_choice_scope": "partial",
            "compiled_binary_identity_scope": "partial",
            "fixed_single_stack_only": True,
            "local_dry_run_is_scientific_evidence": False,
            "per_fault_outcomes_only": True,
            "population_detection_rate_allowed": False,
        },
        "R33 held-out formal claim boundary",
    )
    require(
        formal_protocol.get("fault_ids") == [row[0] for row in expected_faults],
        "R33 held-out formal fault order drift",
    )
    expected_fault_bindings = {
        fault_id: {
            "fault_id": fault_id,
            "rank": rank,
            "fault_definition_sha256": definition_sha256,
            "expected_primary_gate": expected_gate,
        }
        for rank, (fault_id, expected_gate, definition_sha256) in enumerate(expected_faults)
    }
    require(
        formal_protocol.get("fault_bindings") == expected_fault_bindings,
        "R33 held-out formal fault binding drift",
    )
    r33_source_bindings = {
        "r33_aggregate_source_sha256": ("r33_aggregate_fresh_faults.py", "991c05ea6cbedf5ffe5c8a56f26075d0ae7fcadadbc37c22d0d7397861731447"),
        "r33_core_source_sha256": ("r33_executor_core.py", "2ba327000bfeafb0b62be4a1a74ec7991fe763f684b213768b2657b6ecd87689"),
        "r33_core_test_source_sha256": ("r33_test_executor_core.py", "71b2a6b982c22fd26201f7c453085522753766c157e6a0ed7469251442e0b419"),
        "r33_executor_source_sha256": ("r33_execute_fresh_faults.py", "f12f8f4af274dcbb964d2f5a829e3ec68cfc2c664d8dd899c3c3a5a64eccf562"),
        "r33_launcher_source_sha256": ("r33_launch_fresh_faults.sh", "ffa9d0e2659df28d30d83b10c502fdeb763124316bef6f36561672ce49c05f93"),
        "r33_mapping_test_source_sha256": ("r33_test_fault_mapping.py", "b8f8a5b699afb919ad2f93a6beecc744d90b48a91b7b87832e1c5f65c38bc7d0"),
        "r33_prepare_source_sha256": ("r33_prepare_fresh_faults.py", "b581e8b15629657bb2d0d320e86df901df0b1c7e778e908dda28f76fdf156d44"),
        "r33_replay_source_sha256": ("r33_fault_replay.py", "cdab05bb4d5a1b93b3d6c9fe78f8eb5bde2e7a629c326e8b159153f4c7fc32f6"),
    }
    protocol_sources = formal_protocol.get("source_bindings", {})
    for key, (filename, expected) in r33_source_bindings.items():
        require(protocol_sources.get(key) == expected, f"R33 held-out source binding drift: {key}")
        require_sha256(PAPER / "scripts" / filename, expected, f"R33 held-out source {filename}")

    amendment = " ".join(R33_HELDOUT_AMENDMENT.read_text().split())
    for fragment in (
        "before construction of any matched clean case",
        "No fault injection, mutant execution, detector predicate, semantic oracle, or scientific classification was reached.",
        "changes only the warm-up disposal statement",
        "All author-frozen fault definitions",
        "All five ranks are rerun; there is no selective replacement",
    ):
        require(fragment in amendment, f"R33 held-out Attempt-B amendment omits: {fragment}")

    terminal_count = verify_relocated_absolute_sha256_ledger(
        R33_HELDOUT_TERMINAL_LEDGER,
        R33_HELDOUT_RUN_ROOT,
        "R33 held-out terminal ledger",
    )
    require(terminal_count == 208, "R33 held-out terminal file count drift")
    summary = json.loads(R33_HELDOUT_SUMMARY.read_text())
    require_exact_fields(
        summary,
        {
            "schema_version": "forkaudit-r33-five-pair-summary-v1",
            "run_id": run_id,
            "status": "completed_strict_scientific_aggregation",
            "scientific_valid": True,
            "pair_count": 5,
            "clean_gate_pass_count": 5,
            "caught_by_expected_primary_gate_count": 5,
            "escaped_expected_primary_gate_count": 0,
            "operational_invalid_count": 0,
            "missing_pair_count": 0,
            "negative_or_escape_retained": True,
            "population_detection_rate_computed": False,
            "claim_boundary": {
                "candidate_import_free_replay_is_not_independent_live_recapture": True,
                "heldout_population_claim_allowed": False,
                "per_frozen_fault_outcomes_only": True,
                "single_model_single_stack_fixed_case_only": True,
            },
        },
        "R33 held-out strict aggregate",
    )
    summary_rows = summary.get("rows")
    require(isinstance(summary_rows, list) and len(summary_rows) == 5, "R33 held-out summary-row count drift")
    semantic_expectations = [
        {"generated_tokens_exact": True, "call_cardinality_comparable": True, "full_fp32_logits_byte_exact": True, "terminal_logical_kv_exact": True, "terminal_gdn_exact": True},
        {"generated_tokens_exact": True, "call_cardinality_comparable": True, "full_fp32_logits_byte_exact": True, "terminal_logical_kv_exact": True, "terminal_gdn_exact": True},
        {"generated_tokens_exact": True, "call_cardinality_comparable": False, "full_fp32_logits_byte_exact": False, "terminal_logical_kv_exact": False, "terminal_gdn_exact": False},
        {"generated_tokens_exact": False, "call_cardinality_comparable": True, "full_fp32_logits_byte_exact": False, "terminal_logical_kv_exact": False, "terminal_gdn_exact": False},
        {"generated_tokens_exact": True, "call_cardinality_comparable": True, "full_fp32_logits_byte_exact": True, "terminal_logical_kv_exact": True, "terminal_gdn_exact": True},
    ]
    observed_semantics: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="forkaudit-r33-heldout-replay-") as temp_dir:
        temporary_root = Path(temp_dir)
        for rank, ((fault_id, expected_gate, definition_sha256), expected_semantic) in enumerate(
            zip(expected_faults, semantic_expectations)
        ):
            rank_root = R33_HELDOUT_RUN_ROOT / f"rank-run-{rank}"
            case_root = rank_root / f"rank-{rank}"
            clean_path = case_root / "clean-case.json"
            mutant_path = case_root / "mutant-case.json"
            pair_path = case_root / "pair.json"
            clean_replay_path = case_root / "clean-gate-replay.json"
            pair_replay_path = case_root / "pair-replay.json"
            clean = json.loads(clean_path.read_text())
            mutant = json.loads(mutant_path.read_text())
            pair = json.loads(pair_path.read_text())
            clean_replay = json.loads(clean_replay_path.read_text())
            pair_replay = json.loads(pair_replay_path.read_text())
            for lane, case in (("clean", clean), ("mutant", mutant)):
                require_exact_fields(
                    case,
                    {
                        "schema_version": "forkaudit-r33-executed-case-v1",
                        "run_id": run_id,
                        "fault_id": fault_id,
                        "rank": rank,
                        "lane": lane,
                        "status": "full_horizon_completed",
                        "operational_invalid": None,
                        "all_existing_gates_enabled": True,
                        "mandatory_coverage_complete": True,
                        "byte_binding_passed": True,
                    },
                    f"R33 held-out {fault_id} {lane} case",
                )
                require_exact_fields(
                    case.get("cleanup", {}),
                    {
                        "completed": True,
                        "registered_backend_restored": True,
                        "strong_references_released": True,
                        "gc_collect_completed": True,
                        "accelerator_cache_cleanup_completed": True,
                        "accelerator_synchronize_completed": True,
                        "allocator_baseline_exact": True,
                        "cleanup_error": None,
                    },
                    f"R33 held-out {fault_id} {lane} cleanup",
                )
            require_exact_fields(
                pair,
                {
                    "schema_version": "forkaudit-r33-rank-pair-v1",
                    "run_id": run_id,
                    "rank": rank,
                    "fault_id": fault_id,
                    "fault_definition_sha256": definition_sha256,
                    "expected_primary_gate": expected_gate,
                    "classification": "caught_by_expected_primary_gate",
                    "first_failed_predicate": expected_gate,
                    "negative_or_escape_retained": True,
                    "status": "completed",
                },
                f"R33 held-out {fault_id} pair",
            )
            pair_references = {
                "clean_case": clean_path,
                "mutant_case": mutant_path,
                "clean_gate_replay": clean_replay_path,
                "pair_replay": pair_replay_path,
            }
            for key, referenced_path in pair_references.items():
                reference = pair.get(key, {})
                require(
                    rank_root / str(reference.get("path", "")) == referenced_path
                    and reference.get("sha256") == sha256(referenced_path),
                    f"R33 held-out {fault_id} {key} byte binding drift",
                )
            require_exact_fields(
                clean_replay,
                {
                    "schema_version": "forkaudit-r33-detached-clean-gate-v1",
                    "fault_id": fault_id,
                    "expected_primary_gate": expected_gate,
                    "status": "clean_gate_passed",
                    "candidate_modules_imported": False,
                    "fault_module_loaded": False,
                    "mutant_authorized": True,
                    "clean_logit_sidecars_verified": 16,
                },
                f"R33 held-out {fault_id} clean replay",
            )
            expected_mutant_sidecars = 17 if rank == 2 else 16
            require_exact_fields(
                pair_replay,
                {
                    "schema_version": "forkaudit-r33-detached-pair-replay-v1",
                    "fault_id": fault_id,
                    "expected_primary_gate": expected_gate,
                    "first_failed_predicate": expected_gate,
                    "classification": "caught_by_expected_primary_gate",
                    "clean_gate_passed": True,
                    "target_predicate_failed": True,
                    "earlier_unrelated_predicates_passed": True,
                    "injection_witness_passed": True,
                    "candidate_modules_imported": False,
                    "negative_or_escape_retained": True,
                    "clean_logit_sidecars_verified": 16,
                    "mutant_logit_sidecars_verified": expected_mutant_sidecars,
                },
                f"R33 held-out {fault_id} pair replay",
            )
            semantic = mutant.get("semantic_comparison_to_clean", {})
            require_exact_fields(
                semantic,
                {**expected_semantic, "secondary_only": True},
                f"R33 held-out {fault_id} secondary semantics",
            )
            observed_semantics.append({"fault_id": fault_id, **expected_semantic})
            summary_row = summary_rows[rank]
            require_exact_fields(
                summary_row,
                {
                    "rank": rank,
                    "fault_id": fault_id,
                    "fault_definition_sha256": definition_sha256,
                    "expected_primary_gate": expected_gate,
                    "classification": "caught_by_expected_primary_gate",
                    "first_failed_predicate": expected_gate,
                    "clean_case_sha256": sha256(clean_path),
                    "mutant_case_sha256": sha256(mutant_path),
                    "clean_gate_passed": True,
                    "byte_binding_passed": True,
                    "operational_invalid": False,
                },
                f"R33 held-out {fault_id} aggregate row",
            )
            fresh_clean_output = temporary_root / f"rank-{rank}-clean.json"
            fresh_pair_output = temporary_root / f"rank-{rank}-pair.json"
            run_json_replay(
                [
                    sys.executable, str(PAPER / "scripts/r33_fault_replay.py"),
                    "--mode", "clean", "--fault-id", fault_id,
                    "--artifact-root", str(rank_root),
                    "--clean-case", str(clean_path),
                    "--output", str(fresh_clean_output),
                ],
                clean_replay_path,
                f"R33 held-out fresh clean replay {fault_id}",
            )
            run_json_replay(
                [
                    sys.executable, str(PAPER / "scripts/r33_fault_replay.py"),
                    "--mode", "pair", "--fault-id", fault_id,
                    "--artifact-root", str(rank_root),
                    "--clean-case", str(clean_path),
                    "--mutant-case", str(mutant_path),
                    "--fault-definition-sha256", definition_sha256,
                    "--output", str(fresh_pair_output),
                ],
                pair_replay_path,
                f"R33 held-out fresh pair replay {fault_id}",
            )
        fresh_summary_output = temporary_root / "summary.json"
        run_json_replay(
            [
                sys.executable, str(PAPER / "scripts/r33_aggregate_fresh_faults.py"),
                "--protocol", str(R33_HELDOUT_FORMAL_PROTOCOL),
                "--expected-protocol-sha256", expected_sha256[R33_HELDOUT_FORMAL_PROTOCOL],
                "--rank-run-root", str(R33_HELDOUT_RUN_ROOT),
                "--output", str(fresh_summary_output),
            ],
            R33_HELDOUT_SUMMARY,
            "R33 held-out fresh strict aggregation",
        )
    require(
        sum(bool(row["generated_tokens_exact"]) for row in observed_semantics) == 4
        and sum(bool(row["call_cardinality_comparable"]) for row in observed_semantics) == 4
        and sum(
            bool(row["call_cardinality_comparable"])
            and bool(row["full_fp32_logits_byte_exact"])
            for row in observed_semantics
        ) == 3,
        "R33 held-out secondary semantic denominator drift",
    )

    verification = json.loads(R33_HELDOUT_RESULT_VERIFICATION.read_text())
    require_exact_fields(
        verification,
        {
            "schema_version": "forkaudit-r33-local-result-verification-v1",
            "run_id": run_id,
            "trial_id": 1_900_821,
            "archive": {
                "path": "r33-fresh-faults-20260825b-result.tar.gz",
                "sha256": expected_sha256[R33_HELDOUT_RESULT_ARCHIVE],
            },
            "summary": {
                "path": "r33-fresh-faults-20260825b/summary.json",
                "sha256": expected_sha256[R33_HELDOUT_SUMMARY],
                "scientific_valid": True,
                "pair_count": 5,
                "clean_gate_pass_count": 5,
                "caught_by_expected_primary_gate_count": 5,
                "escaped_expected_primary_gate_count": 0,
                "operational_invalid_count": 0,
                "population_detection_rate_computed": False,
            },
            "claim_boundary": {
                "per_frozen_fault_outcomes_only": True,
                "heldout_population_claim_allowed": False,
                "single_model_single_stack_fixed_case_only": True,
                "candidate_import_free_replay_is_not_independent_live_recapture": True,
            },
        },
        "R33 held-out local result verification",
    )
    require_exact_fields(
        verification.get("terminal_ledger", {}),
        {
            "path": "r33-fresh-faults-20260825b/terminal-files.sha256",
            "sha256": expected_sha256[R33_HELDOUT_TERMINAL_LEDGER],
            "verified_file_count": 208,
            "all_entries_verified": True,
        },
        "R33 held-out local terminal verification",
    )

    expected_registry_outcomes = [
        {
            "fault_id": "HF01_DELAYED_TAIL_DETACH", "first_failed_predicate": "TAIL_COPY_BEFORE_FIRST_APPEND_WRITE",
            "tokens_exact": True, "full_fp32_logits_byte_exact": True,
            "terminal_logical_kv_exact": True, "terminal_gdn_exact": True,
        },
        {
            "fault_id": "HF02_INACTIVE_DOCUMENT_LANE_SCRIBBLE", "first_failed_predicate": "PHYSICAL_DOCUMENT_PREFIX_IMMUTABLE",
            "tokens_exact": True, "full_fp32_logits_byte_exact": True,
            "terminal_logical_kv_exact": True, "terminal_gdn_exact": True,
        },
        {
            "fault_id": "HF03_DUPLICATE_COMMITTED_DISPATCH", "first_failed_predicate": "ORDERED_CALL_CARDINALITY",
            "tokens_exact": True, "full_fp32_logits_comparable": False,
            "clean_call_count": 16, "mutant_call_count": 17,
            "terminal_logical_kv_exact": False, "terminal_gdn_exact": False,
        },
        {
            "fault_id": "HF04_EFFECTIVE_SCALE_DRIFT", "first_failed_predicate": "ATTENTION_EFFECTIVE_SCALE",
            "tokens_exact": False, "full_fp32_logits_byte_exact": False,
            "terminal_logical_kv_exact": False, "terminal_gdn_exact": False,
        },
        {
            "fault_id": "HF05_STALE_GDN_BINDING_TOKEN_AFTER_REBIND", "first_failed_predicate": "GDN_COMPLETED_BINDING_TOKEN_ADVANCE",
            "tokens_exact": True, "full_fp32_logits_byte_exact": True,
            "terminal_logical_kv_exact": True, "terminal_gdn_exact": True,
        },
    ]
    registry = experiments.get(evidence_id, {})
    require_exact_fields(
        registry,
        {
            "source_experiment_id": run_id,
            "status": "verified_bounded_pdf_only_fresh_heldout_five_pair_outcomes",
            "active_manuscript_support": True,
            "fault_outcomes": expected_registry_outcomes,
        },
        "R33 held-out registry identity/outcomes",
    )
    require_exact_fields(
        registry.get("source_binding", {}),
        {
            "author_faults_raw_sha256": expected_sha256[R33_HELDOUT_FAULTS],
            "author_protocol_raw_sha256": expected_sha256[R33_HELDOUT_AUTHOR_PROTOCOL],
            "author_manifest_sha256": expected_sha256[R33_HELDOUT_AUTHOR_MANIFEST],
            "designer_input_pdf_sha256": expected_sha256[R33_HELDOUT_DESIGNER_PDF],
            "attempt_b_amendment_sha256": expected_sha256[R33_HELDOUT_AMENDMENT],
            "formal_protocol_sha256": expected_sha256[R33_HELDOUT_FORMAL_PROTOCOL],
            "execution_package_sha256": expected_sha256[R33_HELDOUT_EXECUTION_PACKAGE],
            "result_archive_sha256": expected_sha256[R33_HELDOUT_RESULT_ARCHIVE],
            "local_result_verification_sha256": expected_sha256[R33_HELDOUT_RESULT_VERIFICATION],
            "summary_sha256": expected_sha256[R33_HELDOUT_SUMMARY],
            "terminal_ledger_sha256": expected_sha256[R33_HELDOUT_TERMINAL_LEDGER],
        },
        "R33 held-out registry source binding",
    )
    require_exact_fields(
        registry.get("validation", {}),
        {
            "scientific_run_valid": True,
            "trial_id": 1_900_821,
            "fresh_fault_pairs": 5,
            "clean_gate_pass_count": 5,
            "mutants_caught_by_frozen_expected_primary_gate": 5,
            "escaped_expected_primary_gate_count": 0,
            "operational_invalid_count": 0,
            "terminal_files_verified": 208,
            "mutants_with_exact_surfaced_tokens": 4,
            "comparable_logit_pairs": 4,
            "comparable_pairs_with_exact_full_fp32_logits": 3,
            "all_clean_and_mutant_allocator_baselines_exact": True,
            "population_detection_rate_computed": False,
        },
        "R33 held-out registry validation",
    )
    require_exact_fields(
        registry.get("attempt_history", {}),
        {
            "attempt_a_scientific_outcome": None,
            "attempt_a_status": "operational_invalid_before_any_clean_case_construction",
            "author_faults_or_scientific_gates_changed": False,
            "all_five_ranks_rerun_nonselectively": True,
            "new_non_overwriting_output_root_used": True,
        },
        "R33 held-out registry attempt history",
    )
    require(
        "not a population detection rate" in registry.get("replay_boundary", ""),
        "R33 held-out registry omits no-population-rate boundary",
    )

    integrated_heldout = integrated.get("pdf_only_fresh_heldout_faults", {})
    require_exact_fields(
        integrated_heldout,
        {
            "evidence_id": evidence_id,
            "run_id": run_id,
            "status": "verified_bounded_pdf_only_fresh_heldout_five_pair_outcomes",
            "active_manuscript_support": True,
            "designer_input_pdf_path": "evidence/r33_fresh_faults/author_freeze/designer_input/round32_input.pdf",
            "designer_input_pdf_sha256": expected_sha256[R33_HELDOUT_DESIGNER_PDF],
            "author_faults_raw_sha256": expected_sha256[R33_HELDOUT_FAULTS],
            "author_protocol_raw_sha256": expected_sha256[R33_HELDOUT_AUTHOR_PROTOCOL],
            "formal_protocol_sha256": expected_sha256[R33_HELDOUT_FORMAL_PROTOCOL],
            "execution_package_sha256": expected_sha256[R33_HELDOUT_EXECUTION_PACKAGE],
            "result_archive_sha256": expected_sha256[R33_HELDOUT_RESULT_ARCHIVE],
            "summary_sha256": expected_sha256[R33_HELDOUT_SUMMARY],
            "terminal_ledger_sha256": expected_sha256[R33_HELDOUT_TERMINAL_LEDGER],
            "terminal_files_verified": 208,
            "scientific_run_valid": True,
            "fault_pairs": 5,
            "clean_gate_pass_count": 5,
            "caught_by_frozen_expected_primary_gate_count": 5,
            "escaped_expected_primary_gate_count": 0,
            "operational_invalid_count": 0,
            "mutants_with_exact_surfaced_tokens": 4,
            "comparable_logit_pairs": 4,
            "comparable_pairs_with_exact_full_fp32_logits": 3,
            "population_detection_rate_computed": False,
        },
        "R33 integrated held-out support",
    )
    require_exact_fields(
        integrated_heldout.get("attempt_a", {}),
        {
            "status": "operational_invalid_before_clean_case_construction",
            "scientific_outcome": None,
            "fault_or_gate_changed_for_attempt_b": False,
            "all_ranks_rerun": True,
        },
        "R33 integrated held-out Attempt-A boundary",
    )
    authorized_claim = str(integrated_heldout.get("authorized_claim", ""))
    for fragment in (
        "received only the prior PDF", "five new all-gates-on fault pairs",
        "all five matched clean gates passed", "every mutant was rejected first",
        "four mutants preserved the surfaced token sequence", "HF01, HF02, and HF05",
    ):
        require(fragment in authorized_claim, f"R33 held-out authorized claim omits {fragment}")
    integrated_boundary = str(integrated_heldout.get("boundary", ""))
    for fragment in (
        "Not a population detection rate", "natural-defect study",
        "false-negative-rate estimate", "fault-set-completeness result",
        "cross-model/runtime/hardware generality claim",
    ):
        require(fragment in integrated_boundary, f"R33 integrated held-out boundary omits {fragment}")

    heldout_claim = claims.get("C-R33-PDF-ONLY-FRESH-HELDOUT-01", {})
    require_exact_fields(
        heldout_claim,
        {"status": "verified_bounded_active", "evidence_ids": evidence_id},
        "R33 held-out claim-map identity",
    )
    heldout_claim_text = " ".join(heldout_claim.values())
    for fragment in (
        "prior 24-page PDF", "all five matched clean gates pass",
        "every mutant is rejected first", "four mutants preserve surfaced token equality",
        "HF01, HF02, and HF05", "208 terminal files", "not a detection rate",
    ):
        require(fragment in heldout_claim_text, f"R33 held-out claim map omits {fragment}")
    rejected_heldout = claims.get("C-UNSUPPORTED-R33-HELDOUT-EXPANSION", {})
    require_exact_fields(
        rejected_heldout,
        {"status": "rejected", "evidence_ids": evidence_id},
        "R33 rejected held-out expansion",
    )
    require(
        "population detection rate" in rejected_heldout.get("claim", "")
        and "without pooling" in rejected_heldout.get("extraction_or_test", ""),
        "R33 rejected held-out expansion semantics drift",
    )

    method_expectations = {
        "M-R33-HELDOUT-AUTHOR-FREEZE": (
            "round32_input.pdf",
            "prior 24-page PDF",
        ),
        "M-R33-HELDOUT-EXECUTION": (
            "r33_execute_fresh_faults.py",
            "Attempt B changes only discarded-warmup alias cleanup",
        ),
        "M-R33-HELDOUT-REPLAY": (
            "r33_fault_replay.py",
            "5/5 mutants fail first at the frozen primary gate",
        ),
    }
    for method_id, (source_fragment, configuration_fragment) in method_expectations.items():
        row = methods.get(method_id, {})
        require(source_fragment in row.get("source_path", ""), f"{method_id} source provenance drift")
        require(configuration_fragment in row.get("configuration_or_runtime", ""), f"{method_id} configuration provenance drift")
        require(row.get("manuscript_locations", "") not in ("", "N/A"), f"{method_id} manuscript provenance missing")

    return {
        "evidence_id": evidence_id,
        "run_id": run_id,
        "designer_input_pdf_sha256": expected_sha256[R33_HELDOUT_DESIGNER_PDF],
        "formal_protocol_sha256": expected_sha256[R33_HELDOUT_FORMAL_PROTOCOL],
        "result_archive_sha256": expected_sha256[R33_HELDOUT_RESULT_ARCHIVE],
        "summary_sha256": expected_sha256[R33_HELDOUT_SUMMARY],
        "terminal_ledger_sha256": expected_sha256[R33_HELDOUT_TERMINAL_LEDGER],
        "terminal_files_verified": terminal_count,
        "fresh_clean_replays_exact": 5,
        "fresh_pair_replays_exact": 5,
        "fresh_strict_aggregation_exact": True,
        "clean_gate_pass_count": 5,
        "caught_by_frozen_expected_primary_gate_count": 5,
        "escaped_expected_primary_gate_count": 0,
        "operational_invalid_count": 0,
        "mutants_with_exact_surfaced_tokens": 4,
        "comparable_logit_pairs": 4,
        "comparable_pairs_with_exact_full_fp32_logits": 3,
        "population_detection_rate_computed": False,
        "per_frozen_fault_outcomes_only": True,
        "fault_outcomes": expected_registry_outcomes,
    }


def validate_r35_historical_alias_evidence(
    integrated: dict[str, object],
    experiments: dict[str, dict[str, object]],
    claims: dict[str, dict[str, str]],
    methods: dict[str, dict[str, str]],
) -> dict[str, object]:
    """Re-execute and bind the bounded historical-regression evidence."""

    evidence_id = "E-R35-HISTORICAL-ALIAS-REGRESSION-A"
    run_id = "R35-HISTORICAL-ALIAS-20260826A"
    expected_sha256 = {
        R35_DESIGN_DECISION: "4f83e5ccf7cd70ac858dc63b30450411e6a4c536edcec3488296befe6db8b84e",
        R35_PREREGISTRATION: "6cffd4775a0969007b504adc4a091b8587f17d2f5317734e683fba242e598d74",
        R35_STATIC_EXECUTION_INPUT: "26be505aad84adbab18b3752928e31f391912661f116c4fe7a9c8a4f7dd338fa",
        R35_SOURCE_LEDGER: "672d847665519976a7ddf3d88ff969570723b7159b43cf02b8b6037862659051",
        R35_FREEZE_RECEIPT: "e0bc2c335b197ca3f204aa039b4d218308bb5fa3e55a176d9a24d920d22e8c2f",
        R35_RESOURCE_AMENDMENT: "a48557e64994c6a70615f8851e2c9b2e19ba44da2c7417fb331a860e3e58aad1",
        R35_EXECUTION_PACKAGE: "cc979b249c6b6e214d377d377cb5d249e1441f78e469b179515e2ed0cfb174cf",
        R35_RESULT_ARCHIVE: "93e177c7cb483aa3c1f02ec7e602f8af2fbc33fefe8deee45189af0963e4317d",
        R35_RESULT_VERIFICATION: "4f04f6fdd630042aac76b9d23877bb3849664a42ea8c6edac51944fab58bd765",
        R35_AGGREGATE: "7d23418037c718feb2e7170667dd7bc069025bbaff4b69f64326e040439a8d4e",
        R35_LAUNCH_COMPLETION: "cce3dfd90c63466e425e81582fb692b5459946362762b7ba28be94a497976a22",
        R35_VALIDATOR: "40cdf4d68ab8aba20671dd0cf5fe2053dfe916f5436e0fd53c7a90e437db3a89",
    }
    for path, expected in expected_sha256.items():
        require_sha256(path, expected, f"R35 artifact {path.relative_to(PAPER)}")

    completed = subprocess.run(
        [
            sys.executable,
            str(R35_VALIDATOR),
            "--paper-root",
            str(PAPER),
            "--evidence-root",
            str(R35_ROOT),
            "--formal-root",
            str(R35_FORMAL_ROOT),
            "--output",
            str(R35_RESULT_VERIFICATION),
        ],
        cwd=PAPER,
        check=False,
        text=True,
        capture_output=True,
        timeout=180,
    )
    require(
        completed.returncode == 0,
        "R35 fail-closed verifier failed: "
        f"stdout={completed.stdout!r}, stderr={completed.stderr!r}",
    )
    require_sha256(
        R35_RESULT_VERIFICATION,
        expected_sha256[R35_RESULT_VERIFICATION],
        "R35 deterministic verification receipt after re-execution",
    )

    verification = json.loads(R35_RESULT_VERIFICATION.read_text())
    require_exact_fields(
        verification,
        {
            "schema_version": "forkaudit-r35-local-evidence-verification-v1",
            "status": "verified_fail_closed",
            "operational_valid": True,
            "run_id": run_id,
        },
        "R35 verification identity",
    )
    require_exact_fields(
        verification.get("validator", {}),
        {
            "path": "scripts/validate_r35_historical_alias_evidence.py",
            "sha256": expected_sha256[R35_VALIDATOR],
            "standard_library_only": True,
            "postexecution_verifier_not_in_frozen_execution_package": True,
        },
        "R35 verifier receipt",
    )
    require_exact_fields(
        verification.get("evidence_graph", {}),
        {
            "launch_completion_sha256": expected_sha256[R35_LAUNCH_COMPLETION],
            "aggregate_sha256": expected_sha256[R35_AGGREGATE],
            "launch_artifact_count": 9,
            "rank_result_count": 8,
            "rank_replay_count": 8,
            "lane_count": 24,
            "fp32_sidecar_count": 24,
            "unique_case_nonce_count": 24,
            "unique_gpu_uuid_count": 8,
            "all_hash_edges_verified": True,
            "all_sidecars_size_shape_finite_verified": True,
            "all_rank_status_uuid_fault_isolation_verified": True,
        },
        "R35 evidence graph",
    )
    require_exact_fields(
        verification.get("detached_reexecution", {}),
        {
            "frozen_replay_program_sha256": "3e5ef9be9d27161460a25f82514c135676d389b41439e70c095efaee0a7ce259",
            "frozen_aggregate_program_sha256": "096e3005aa1701e60310050a685368c38bf8f3046467874c35f52c795e6e29f8",
            "rank_replay_count": 8,
            "rank_replay_bytes_exact": True,
            "aggregate_bytes_exact": True,
            "aggregate_sha256": expected_sha256[R35_AGGREGATE],
            "candidate_or_runtime_modules_imported_by_verifier": False,
        },
        "R35 detached re-execution",
    )
    require(
        verification.get("headline_outcomes")
        == {
            "all_eight_historical_expected_gates_reproduced": True,
            "all_eight_historical_output_only_pairs_exact": True,
            "all_eight_materialized_storage_contracts_clean": True,
            "all_eight_repaired_pairs_semantic_and_terminal_exact": True,
            "all_eight_repaired_storage_contracts_clean": True,
            "registered_positive_headline_rule_satisfied": True,
        },
        "R35 registered headline drift",
    )
    require(
        verification.get("claim_boundary")
        == {
            "archived_and_additional_coordinates_reported_separately": True,
            "eight_rows_are_coordinate_cells_not_eight_natural_bugs": True,
            "normalized_storage_ids_compared_across_lanes": False,
            "normalized_storage_ids_compared_within_lane_only": True,
            "population_detection_rate_computed": False,
            "statistical_independence_claimed": False,
        },
        "R35 claim boundary drift",
    )
    coordinates = verification.get("coordinate_cell_counts", {})
    expected_coordinate_counts = {
        "archived_coordinates": (3, [0, 1, 2]),
        "additional_frozen_inputs": (5, [3, 4, 5, 6, 7]),
        "all_coordinates": (8, list(range(8))),
    }
    for name, (count, ranks) in expected_coordinate_counts.items():
        row = coordinates.get(name, {})
        require_exact_fields(
            row,
            {
                "cell_count": count,
                "ranks": ranks,
                "operational_valid_rank_count": count,
                "all_cells_operationally_valid": True,
                "historical_authenticated_receipt_and_inner_predicate_count": count,
                "historical_base_immutability_violation_count": count,
                "repaired_audit_pass_count": count,
                "materialized_audit_pass_count": count,
                "registered_positive_headline_satisfied": True,
            },
            f"R35 {name} coordinate counts",
        )
        historical = row.get("historical_vs_materialized_true_counts", {})
        repaired = row.get("repaired_vs_materialized_true_counts", {})
        for metric in (
            "greedy_token_exact",
            "full_fp32_logits_exact",
            "request0_terminal_gdn_content_exact",
            "logical_kv_content_exact",
        ):
            require(historical.get(metric) == count, f"R35 {name} historical {metric}")
            require(repaired.get(metric) == count, f"R35 {name} repaired {metric}")
        require(
            historical.get("persistent_base_content_only_invariant") == 0
            and repaired.get("persistent_base_content_only_invariant") == count,
            f"R35 {name} persistent-base outcome",
        )

    registry = experiments.get(evidence_id, {})
    require_exact_fields(
        registry,
        {
            "source_experiment_id": run_id,
            "status": "verified_bounded_historical_regression_and_exact_repair",
            "active_manuscript_support": True,
        },
        "R35 registry identity",
    )
    expected_source_binding = {
        "design_decision_sha256": expected_sha256[R35_DESIGN_DECISION],
        "preregistration_sha256": expected_sha256[R35_PREREGISTRATION],
        "static_execution_input_sha256": expected_sha256[R35_STATIC_EXECUTION_INPUT],
        "source_ledger_sha256": expected_sha256[R35_SOURCE_LEDGER],
        "preexecution_freeze_receipt_sha256": expected_sha256[R35_FREEZE_RECEIPT],
        "resource_amendment_sha256": expected_sha256[R35_RESOURCE_AMENDMENT],
        "execution_package_sha256": expected_sha256[R35_EXECUTION_PACKAGE],
        "result_archive_sha256": expected_sha256[R35_RESULT_ARCHIVE],
        "local_result_verification_sha256": expected_sha256[R35_RESULT_VERIFICATION],
        "local_result_verifier_sha256": expected_sha256[R35_VALIDATOR],
        "aggregate_sha256": expected_sha256[R35_AGGREGATE],
        "launch_completion_sha256": expected_sha256[R35_LAUNCH_COMPLETION],
    }
    require(
        registry.get("source_binding") == expected_source_binding,
        "R35 registry source binding drift",
    )
    require_exact_fields(
        registry.get("validation", {}),
        {
            "scientific_run_valid": True,
            "job_id": 251492,
            "trial_id": 1905906,
            "fixed_model": "Qwen/Qwen3.5-35B-A3B",
            "hardware": "eight NVIDIA H20-3e GPUs, one rank per GPU",
            "rank_results": 8,
            "detached_replays": 8,
            "lanes": 24,
            "fp32_sidecar_files_verified": 24,
            "fp32_sidecar_bytes_each": 993280,
            "unique_case_nonces": 24,
            "archived_coordinate_cells": 3,
            "additional_frozen_input_cells": 5,
            "historical_expected_gate_cells": 8,
            "historical_output_and_terminal_state_exact_cells": 8,
            "historical_persistent_base_invariant_passes": 0,
            "repaired_storage_clean_cells": 8,
            "repaired_vs_materialized_exact_cells": 8,
            "materialized_storage_clean_cells": 8,
            "local_replay_byte_identical_ranks": 8,
            "local_aggregate_byte_identical": True,
            "mutation_requested": False,
            "mutation_applied": False,
            "population_detection_rate_computed": False,
            "statistical_independence_claimed": False,
        },
        "R35 registry validation",
    )
    artifact_paths = registry.get("artifact_paths", [])
    require(isinstance(artifact_paths, list) and len(artifact_paths) == 13, "R35 artifact-path count")
    for relative in artifact_paths:
        require(isinstance(relative, str) and (PAPER / relative).is_file(), f"R35 missing registry artifact {relative}")
    artifact_sets = registry.get("artifact_sets", [])
    require(isinstance(artifact_sets, list) and len(artifact_sets) == 3, "R35 artifact-set count")
    for artifact_set in artifact_sets:
        require(
            len(list(PAPER.glob(str(artifact_set.get("glob", "")))))
            == artifact_set.get("expected_count"),
            f"R35 artifact-set drift: {artifact_set}",
        )
    require(len(registry.get("prohibited_expansions", [])) == 6, "R35 prohibited-expansion count")

    integrated_r35 = integrated.get("historical_alias_regression", {})
    require_exact_fields(
        integrated_r35,
        {
            "evidence_id": evidence_id,
            "run_id": run_id,
            "status": "verified_bounded_historical_regression_and_exact_repair",
            "active_manuscript_support": True,
            "artifact_root": "evidence/r35_historical_alias_regression",
            **expected_source_binding,
            "scientific_run_valid": True,
            "rank_results": 8,
            "detached_replays": 8,
            "lanes": 24,
            "fp32_sidecars": 24,
            "unique_case_nonces": 24,
            "archived_coordinate_cells": 3,
            "additional_frozen_input_cells": 5,
            "historical_expected_authenticated_first_gate": "gdn_completed_binding_rebound",
            "historical_expected_gate_cells": 8,
            "historical_vs_materialized_exact_tokens": 8,
            "historical_vs_materialized_exact_full_fp32_logits": 8,
            "historical_vs_materialized_exact_terminal_request_gdn": 8,
            "historical_vs_materialized_exact_logical_kv": 8,
            "historical_persistent_base_invariant_passes": 0,
            "repaired_storage_clean_cells": 8,
            "materialized_storage_clean_cells": 8,
            "repaired_vs_materialized_all_semantic_and_terminal_exact_cells": 8,
            "local_detached_replays_byte_identical": 8,
            "local_aggregate_byte_identical": True,
            "mutation_requested": False,
            "mutation_applied": False,
            "population_detection_rate_computed": False,
            "statistical_independence_claimed": False,
        },
        "R35 integrated support",
    )
    for fragment in (
        "one fixed Qwen3.5/H20 stack",
        "not independent defects or a natural-bug corpus",
        "Persistent-base immutability also catches all eight historical cells",
        "earlier owner/layer/family localization",
        "not exclusive detection",
    ):
        require(fragment in str(integrated_r35), f"R35 integrated boundary omits {fragment}")

    active_claim = claims.get("C-R35-HISTORICAL-ALIAS-01", {})
    require_exact_fields(
        active_claim,
        {"status": "verified_bounded_active", "evidence_ids": evidence_id},
        "R35 claim-map identity",
    )
    active_claim_text = " ".join(active_claim.values())
    for fragment in (
        "one previously encountered borrowed-state alias defect",
        "three archived-coordinate and five additional frozen-input cells",
        "persistent-base immutability fails in all eight cells",
        "not independent defects",
        "earlier owner/layer/family localization",
    ):
        require(fragment in active_claim_text, f"R35 claim map omits {fragment}")
    rejected_claim = claims.get("C-UNSUPPORTED-R35-EXPANSION", {})
    require_exact_fields(
        rejected_claim,
        {"status": "rejected", "evidence_ids": evidence_id},
        "R35 rejected expansion",
    )
    require(
        "eight natural bugs" in rejected_claim.get("claim", "")
        and "population_detection_rate_computed=false" in rejected_claim.get("extraction_or_test", ""),
        "R35 rejected expansion semantics drift",
    )
    method_expectations = {
        "M-R35-HISTORICAL-PROVENANCE": "design_decision.md",
        "M-R35-HISTORICAL-EXECUTION": "r35_run_historical_alias_regression.py",
        "M-R35-HISTORICAL-REPLAY": "r35_replay_historical_alias_regression.py",
        "M-R35-HISTORICAL-AGGREGATE": "r35_aggregate_historical_alias_regression.py",
    }
    for method_id, source_fragment in method_expectations.items():
        row = methods.get(method_id, {})
        require(source_fragment in row.get("source_path", ""), f"{method_id} source provenance drift")
        require(row.get("manuscript_locations", "") not in ("", "N/A"), f"{method_id} manuscript provenance missing")

    return {
        "evidence_id": evidence_id,
        "run_id": run_id,
        "result_archive_sha256": expected_sha256[R35_RESULT_ARCHIVE],
        "result_verification_sha256": expected_sha256[R35_RESULT_VERIFICATION],
        "result_verifier_sha256": expected_sha256[R35_VALIDATOR],
        "aggregate_sha256": expected_sha256[R35_AGGREGATE],
        "launch_completion_sha256": expected_sha256[R35_LAUNCH_COMPLETION],
        "rank_results": 8,
        "detached_replays_byte_exact": 8,
        "lanes": 24,
        "fp32_sidecars": 24,
        "unique_case_nonces": 24,
        "archived_coordinate_cells": 3,
        "additional_frozen_input_cells": 5,
        "historical_expected_gate_cells": 8,
        "historical_output_and_terminal_state_exact_cells": 8,
        "historical_persistent_base_invariant_violations": 8,
        "repaired_storage_clean_and_exact_cells": 8,
        "population_detection_rate_computed": False,
        "statistical_independence_claimed": False,
    }


def visible_tex_source(source: str) -> str:
    """Return manuscript source that can contribute to the rendered paper.

    Checking raw source can let non-rendered prose satisfy required-claim
    assertions.  Remove any false-condition blocks and TeX comments before
    text checks.
    """

    previous = None
    visible = source
    while previous != visible:
        previous = visible
        visible = re.sub(r"\\iffalse\b.*?\\fi\b", " ", visible, flags=re.DOTALL)
    uncommented: list[str] = []
    for line in visible.splitlines():
        match = re.search(r"(?<!\\)%", line)
        uncommented.append(line[: match.start()] if match else line)
    return "\n".join(uncommented)


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    manuscript = MANUSCRIPT.read_text()
    visible_manuscript = visible_tex_source(manuscript)
    rendered_table_source = "\n".join(
        visible_tex_source(path.read_text()) for path in RENDERED_TABLE_INPUTS
    )
    normalized_manuscript = " ".join(
        (visible_manuscript + "\n" + rendered_table_source).split()
    )
    gdn_oracle = json.loads(GDN_ORACLE_VALIDATION.read_text())
    serving = json.loads(SERVING_PANEL.read_text())
    hypic = json.loads(HYPIC_SUMMARY.read_text())
    hypic_store = json.loads(HYPIC_STORE_ACCEPTANCE.read_text())
    assurance = json.loads(ASSURANCE_BOUNDARY.read_text())
    integrated = json.loads(INTEGRATED_RESULTS.read_text())
    experiments = indexed_experiment_registry()
    claims = load_tsv_index(
        CLAIM_EVIDENCE_MAP,
        [
            "claim_id",
            "status",
            "claim",
            "evidence_ids",
            "extraction_or_test",
            "scope",
            "manuscript_locations",
        ],
        "claim_id",
        "claim-evidence map",
    )
    methods = load_tsv_index(
        METHOD_PROVENANCE,
        [
            "method_id",
            "method_statement",
            "source_path",
            "symbol_or_lines",
            "configuration_or_runtime",
            "manuscript_locations",
        ],
        "method_id",
        "method provenance",
    )
    require(
        assurance["schema_version"] == "forkaudit-assurance-boundary-derived-v1",
        "assurance-boundary schema drift",
    )
    require(
        integrated["schema_version"]
        == "forkaudit-round-35-integrated-reviewer-summary-v6",
        "integrated-results schema drift",
    )
    require(
        integrated["status"]
        == "verified_bounded_with_explicit_missingness_r33_capture_and_r35_historical_regression",
        "integrated-results status drift",
    )
    r33_capture_audit = validate_r33_capture_evidence(
        integrated, experiments, claims, methods
    )
    r33_heldout_audit = validate_r33_heldout_evidence(
        integrated, experiments, claims, methods
    )
    r35_historical_alias_audit = validate_r35_historical_alias_evidence(
        integrated, experiments, claims, methods
    )
    r29_expected_sha256 = {
        R29_OBSERVER_PREREGISTRATION: "90decffb732d50ec04fbebe5f34d8d5fb7acb0fabec88f1ba6ce53c9e262984c",
        R29_OBSERVER_SOURCE_LEDGER: "9b9a135fff63bbd4bd363d9c2d341e593ddef776d4f07dc0088333baaef91818",
        R29_OBSERVER_AMENDMENT: "641897b76a4a7cf78ea753b05469512b7041b7c226ffb0620534e90751ae9a94",
        R29_OBSERVER_RESULT: "5d8edf442f9dedd1b3e7e2b338a324b2c30f9001df2fcbba5ce4f6ad2f42c0df",
        R29_OBSERVER_REPLAY: "65435c367bab0b22e55c2b79eb91eb3ef1a6ca62ee56b578b93c7920edfd8e29",
        R29_OBSERVER_TERMINAL_LEDGER: "9b39b9c291b173496aa7ca6d52b01fdf3405c0c900f1b95084496d3558b239cc",
        R29_TWO_STREAM_DESIGN: "5c9fc301ec63e2702d097b9d9be9c68758164c653c6c7b53fedad290428a9a96",
        R29_TWO_STREAM_AMENDMENT: "7d03eca752b3a4163168022cac5e5a044a7e03c67d1815899285c47df2578904",
        R29_TWO_STREAM_SOURCE_LEDGER: "4afc29d3d154f66518710a2b3f00a8a4f31d13da2de0e0c425ed91d807489405",
        R29_TWO_STREAM_RESULT: "f110e994536e0d0637109e1b1a76b6d6140626be534f21fd118fb3ba63dce970",
        R29_TWO_STREAM_REPLAY: "6f02c84b832e2b2c4e6c93b9cd0f93c50d9f14e356c655a0c884c791e10a7032",
        R29_TWO_STREAM_TERMINAL_LEDGER: "d8772a6a972a3c178b3b7b59196b44722772e550a71515562e366fcbdd839789",
        R29_LOCAL_OVERHEAD_PREREGISTRATION: "828b2ec619c9c9827e72c0ca1505971afa0bbf678cfd148e424048c4c6e9d406",
        R29_LOCAL_OVERHEAD_RESULT: "2f31b7c4cc996721d5bd332a48a9e253dd4bc872034973f5c71dee759e45c8d7",
        R29_LOCAL_OVERHEAD_VALIDATION: "469e6c7534cec7cf5363c4a0f075003a48a1d645aa0793102806e97c43e35447",
        R29_LOCAL_OVERHEAD_TERMINAL_LEDGER: "5a7edfd60e36b8d34a18153b144028b26e7c69820c5488f8f8f7ee364c787f50",
        R29_LIVE_OVERHEAD_PREREGISTRATION: "2114d2cd85bedc1eafa5d1398fd0afd0d57819c0360c3be3f9ec20f1b2878939",
        R29_LIVE_OVERHEAD_PRESECOND_AMENDMENT: "b93572b35d411cb1b761b322acff622f88aec69cca016244c79e47ab5e6b0d19",
        R29_LIVE_OVERHEAD_REPLAY_AMENDMENT: "faf773740b5a3f920a6192f9bb4cfad15aba9e5b9a8994d14ed6fa01bf223f17",
        R29_LIVE_OVERHEAD_SOURCE_LEDGER: "043efb9139ae579669efda3984071036c46df45971e00ad3692c803ad171d141",
        R29_LIVE_OVERHEAD_RESULT: "3ccf86e2233b560f003d965fdae05a8e3b0773e15976a05c8d70af881338bc22",
        R29_LIVE_OVERHEAD_SEMANTIC_SIDECAR: "1c3e68ffb29ed3567c88b73757507590509c91d82b8641fca213eb39aabeaf07",
        R29_LIVE_OVERHEAD_REPLAY: "7ede1b7edd0a6d0343ebc61c3742264af397922b2910eb4c39c8e658ad94cb4e",
        R29_LIVE_OVERHEAD_TERMINAL_LEDGER: "81808a4b500254dc68145bf54ef748c9bfe1d1f61fd7386e8667f219d9acca3a",
        R29_LIVE_OVERHEAD_STAGE03: "fc7976388859949e75626d07d4a6deb8b18203cc3aad2003b6233bd18c9b33e7",
        R29_LIVE_OVERHEAD_COMPLETED: "73c5b0a824a6388e9122ce0606ab1e29edcc0ccb257b4e6e7a25b04bbcb4ad23",
        R29_HELDOUT_SUITE: "108f2f551838cc261d42ef6697a7dfbce753ce6ac51bc4ce771d1d6b0773b496",
        R29_HELDOUT_AUTHOR_FREEZE: "441206aee8c46c11c7ce1727a83965fc323d107c740448ec96cbec33836fc5ed",
        R29_HELDOUT_AUTHOR_LEDGER: "7c989ee393d08c103eaf7dbfe673f4e2ccdeb933530b5d3b16686f40abfed3c3",
        R29_HELDOUT_EXECUTION_INPUT_V3: "5a522e48650e3010621e6e06c7c8bbab67c074bca1b6a6a0c70aa50133b4e98d",
        R29_HELDOUT_AMENDMENT_V3: "2f51895fd77ed24aede162203fab31f1cc979b44b96330ff880312d5baa0e09a",
        R29_HELDOUT_EXECUTOR_LEDGER_V3: "33ab8469fbfc218839652cede9d0972b328e4ca43236897f9afa1c35cb84f3ce",
        R29_HELDOUT_READINESS_V3: "bf5a91c0c4432d7597cbdad1e4ea64eb343dad2c3409b01f35f74d339ce65e01",
        R29_HELDOUT_FREEZE_V3: "72c131aaad810d62b2a94c5f81c20877c04a12e27e482379a9af6ff65fd55ad4",
    }
    for path, expected in r29_expected_sha256.items():
        require_sha256(path, expected, f"R29 artifact {path.relative_to(PAPER)}")

    r29_observer = json.loads(R29_OBSERVER_RESULT.read_text())
    r29_observer_replay = json.loads(R29_OBSERVER_REPLAY.read_text())
    r29_two_stream = json.loads(R29_TWO_STREAM_RESULT.read_text())
    r29_two_stream_replay = json.loads(R29_TWO_STREAM_REPLAY.read_text())
    r29_local_overhead = json.loads(R29_LOCAL_OVERHEAD_RESULT.read_text())
    r29_local_overhead_validation = json.loads(
        R29_LOCAL_OVERHEAD_VALIDATION.read_text()
    )
    r29_live_overhead = json.loads(R29_LIVE_OVERHEAD_RESULT.read_text())
    r29_live_overhead_replay = json.loads(R29_LIVE_OVERHEAD_REPLAY.read_text())
    r29_heldout_author_freeze = json.loads(R29_HELDOUT_AUTHOR_FREEZE.read_text())
    r29_heldout_readiness = json.loads(R29_HELDOUT_READINESS_V3.read_text())

    r30_expected_sha256 = {
        R30_PREREGISTRATION: "2d47a39a37a57a4c21085697f87e2f3cc8a1c91eeb7b34960dabfa9b8cb51e2b",
        R30_INPUT_MANIFEST: "6b5b92905a6a2b36daf57d38cae86c448225450ed59c19103de4eed7c460b74e",
        R30_PREEXECUTION_PIN: "7ad2b0683d8c13e621296f30df253b10006bb3ae788627686c7af2df0d83e70e",
        R30_SOURCE_LEDGER: "da474e17cc720e413980a45e9e883a0230d1c94e76ab53317829807f10d297a7",
        R30_CAPTURE_MANIFEST: "ff5f9cfb7526b15ecf0b01b2b8309f04d4f6098001f447cd04bfce8d3f8dfd8b",
        R30_RAW_CAPTURE_MANIFEST: "ff5f9cfb7526b15ecf0b01b2b8309f04d4f6098001f447cd04bfce8d3f8dfd8b",
        R30_ORACLE_RESULT: "56473175be6d5803e4512e86cfd01d26b6f61709fc2b2bc97af80525f0e89e41",
        R30_VALIDATION_REPORT: "39aa442b88d357f06f022e51f5336e14dfcb3c51e8a6f90283f4e0285d26850d",
        R30_RAW_ARTIFACTS_LEDGER: "85b90bb9dde6c7dd7800c9302dfb7a4b3e2f7d73b36c83858d61cdf921cdc10f",
        R30_TERMINAL_PRODUCTS_LEDGER: "f125de7cf8c8f4f30a7abb306345559a30de32d083d49ada31f4fd4cb85d4e57",
        R30_REFERENCE: "f9b234406db4226e9b7d96d00fa1165c7ab08ae2a29cfba52f7954fe6cad2e1c",
    }
    for path, expected in r30_expected_sha256.items():
        require_sha256(path, expected, f"R30 artifact {path.relative_to(PAPER)}")

    r30_preregistration = json.loads(R30_PREREGISTRATION.read_text())
    r30_capture = json.loads(R30_CAPTURE_MANIFEST.read_text())
    r30_oracle = json.loads(R30_ORACLE_RESULT.read_text())
    r30_validation = json.loads(R30_VALIDATION_REPORT.read_text())

    require(
        r30_validation.get("schema_version")
        == "forkaudit-r30-expanded-oracle-validation-v1"
        and r30_validation.get("status")
        == "verified_bounded_fully_preregistered"
        and r30_validation.get("candidate_execution_attempts") == 1,
        "R30 validation identity/status drift",
    )
    r30_preexecution = r30_validation.get("preexecution_bindings", {})
    require_exact_fields(
        r30_preexecution,
        {
            "reference_raw_sha256": r30_expected_sha256[R30_REFERENCE],
            "input_manifest_raw_sha256": r30_expected_sha256[R30_INPUT_MANIFEST],
            "preregistration_raw_sha256": r30_expected_sha256[R30_PREREGISTRATION],
            "source_ledger_raw_sha256": r30_expected_sha256[R30_SOURCE_LEDGER],
            "preexecution_pin_raw_sha256": r30_expected_sha256[R30_PREEXECUTION_PIN],
            "source_ledger_replay_passed_before_candidate": True,
            "input_preparation_loaded_or_executed_candidate": False,
            "post_execution_amendment_used": False,
        },
        "R30 preexecution binding",
    )
    require(
        r30_capture.get("schema_version")
        == "forkaudit-r30-expanded-oracle-capture-v1"
        and r30_capture.get("status") == "captured-no-numerical-pass-fields"
        and r30_capture.get("preregistration_raw_sha256")
        == r30_expected_sha256[R30_PREREGISTRATION],
        "R30 capture identity drift",
    )
    require(
        r30_oracle.get("schema_version")
        == "forkaudit-r30-expanded-oracle-reference-v1"
        and r30_oracle.get("candidate_code_imported") is False
        and r30_oracle.get("capture_manifest_raw_sha256")
        == r30_expected_sha256[R30_CAPTURE_MANIFEST]
        and r30_oracle.get("preregistration_raw_sha256")
        == r30_expected_sha256[R30_PREREGISTRATION]
        and r30_oracle.get("reference_source_raw_sha256")
        == r30_expected_sha256[R30_REFERENCE],
        "R30 oracle identity/binding drift",
    )
    r30_coverage = {
        "attention_layers_per_case": 10,
        "attention_query_positions": 160,
        "attention_rows": 20,
        "gdn_layers_per_case": 12,
        "gdn_rows": 24,
        "gdn_token_transitions": 192,
        "input_cases": 2,
    }
    require(
        r30_capture.get("coverage") == r30_coverage
        and r30_oracle.get("coverage") == r30_coverage,
        "R30 coverage drift",
    )
    require(
        len(r30_capture.get("attention_rows", [])) == 20
        and len(r30_capture.get("gdn_rows", [])) == 24
        and len(r30_oracle.get("attention_rows", [])) == 20
        and len(r30_oracle.get("gdn_rows", [])) == 24
        and len(r30_oracle.get("attention_faults", [])) == 20
        and len(r30_oracle.get("gdn_faults", [])) == 24
        and all(
            r30_oracle.get(key) is True
            for key in (
                "all_attention_clean_rows_pass",
                "all_gdn_clean_rows_pass",
                "all_attention_faults_rejected",
                "all_gdn_faults_rejected",
            )
        ),
        "R30 clean/fault decision drift",
    )
    r30_results = r30_validation.get("results", {})
    require_exact_fields(
        r30_results,
        {
            "attention_clean_rows_passed": 20,
            "attention_clean_rows_total": 20,
            "attention_layers_per_case": 10,
            "attention_full_layers_covered": 10,
            "attention_full_layers_total": 10,
            "attention_query_positions": 160,
            "maximum_clean_attention_relative_l2": 0.0018973927068607452,
            "gdn_clean_rows_passed": 24,
            "gdn_clean_rows_total": 24,
            "gdn_layers_per_case": 12,
            "gdn_layers_covered": 12,
            "gdn_layers_total": 30,
            "gdn_token_transitions": 192,
            "maximum_clean_gdn_output_relative_l2": 0.002072614929712376,
            "maximum_clean_gdn_state_relative_l2": 2.0848835241919874e-7,
            "attention_seeded_faults_rejected": 20,
            "attention_seeded_faults_total": 20,
            "attention_fault_counts": {
                "drop_self_key": 5,
                "reverse_kv_tokens": 5,
                "roll_kv_heads": 5,
                "unit_scale": 5,
            },
            "gdn_seeded_faults_rejected": 24,
            "gdn_seeded_faults_total": 24,
            "gdn_fault_counts": {
                "complement_beta": 6,
                "omit_decay": 6,
                "pre_decay_memory": 6,
                "roll_value_heads": 6,
            },
            "all_clean_rows_pass": True,
            "all_seeded_faults_rejected": True,
        },
        "R30 validation result",
    )

    r30_receipts = [
        receipt
        for row in (
            r30_capture.get("attention_rows", [])
            + r30_capture.get("gdn_rows", [])
        )
        for receipt in row.get("arrays", {}).values()
    ]
    require(len(r30_receipts) == 272, "R30 sidecar-receipt count drift")
    r30_relative_paths = [str(receipt.get("relative_path")) for receipt in r30_receipts]
    require(len(set(r30_relative_paths)) == 272, "R30 sidecar paths are not unique")
    r30_sidecar_bytes = 0
    r30_payload_bytes = 0
    for receipt in r30_receipts:
        sidecar = (R30_RAW_CAPTURE_MANIFEST.parent / str(receipt["relative_path"])).resolve()
        try:
            sidecar.relative_to(R30_RAW_CAPTURE_MANIFEST.parent.resolve())
        except ValueError as exc:
            raise RuntimeError(f"R30 unsafe sidecar path: {sidecar}") from exc
        require_sha256(sidecar, str(receipt["sha256"]), "R30 numerical sidecar")
        r30_sidecar_bytes += sidecar.stat().st_size
        r30_payload_bytes += int(receipt["nbytes"])
    require(r30_sidecar_bytes == 140_199_936, "R30 local sidecar-byte drift")
    require(r30_payload_bytes == 140_165_120, "R30 sidecar-payload-byte drift")

    r30_raw_ledger_rows = [
        line for line in R30_RAW_ARTIFACTS_LEDGER.read_text().splitlines() if line
    ]
    require(len(r30_raw_ledger_rows) == 273, "R30 raw-ledger row-count drift")
    for row in r30_raw_ledger_rows:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", row)
        require(match is not None, f"R30 malformed raw-ledger row: {row}")
        expected, remote_path = match.groups()
        if remote_path.endswith("/raw/capture-manifest.json"):
            local_path = R30_RAW_CAPTURE_MANIFEST
        else:
            marker = "/raw/sidecars/"
            require(marker in remote_path, f"R30 unknown raw-ledger path: {remote_path}")
            local_path = R30_RAW_CAPTURE_MANIFEST.parent / "sidecars" / remote_path.split(marker, 1)[1]
        require_sha256(local_path, expected, "R30 raw-ledger artifact")

    r30_terminal = r30_validation.get("terminal_closure", {})
    require_exact_fields(
        r30_terminal,
        {
            "capture_manifest_raw_sha256": r30_expected_sha256[R30_CAPTURE_MANIFEST],
            "oracle_result_raw_sha256": r30_expected_sha256[R30_ORACLE_RESULT],
            "raw_artifacts_ledger_raw_sha256": r30_expected_sha256[
                R30_RAW_ARTIFACTS_LEDGER
            ],
            "terminal_products_ledger_raw_sha256": r30_expected_sha256[
                R30_TERMINAL_PRODUCTS_LEDGER
            ],
            "raw_sidecar_files": 272,
            "raw_sidecar_bytes": 140_199_936,
            "remote_sha256_replay_passed": True,
            "terminal_products_sha256_replay_passed": True,
            "completed_marker_present": True,
        },
        "R30 terminal closure",
    )

    with tempfile.TemporaryDirectory(prefix="forkaudit-r30-replay-") as temp_dir:
        r30_local_result_path = Path(temp_dir) / "oracle-result.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(R30_REFERENCE),
                "--capture-manifest",
                str(R30_RAW_CAPTURE_MANIFEST),
                "--preregistration",
                str(R30_PREREGISTRATION),
                "--output",
                str(r30_local_result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        require(
            completed.returncode == 0 and r30_local_result_path.is_file(),
            f"R30 local numerical replay failed: {completed.stderr[-1000:]}",
        )
        r30_local_replay = json.loads(r30_local_result_path.read_text())
    require(
        r30_local_replay.get("coverage") == r30_coverage
        and r30_local_replay.get("candidate_code_imported") is False
        and all(
            r30_local_replay.get(key) is True
            for key in (
                "all_attention_clean_rows_pass",
                "all_gdn_clean_rows_pass",
                "all_attention_faults_rejected",
                "all_gdn_faults_rejected",
            )
        ),
        "R30 local numerical replay decision drift",
    )
    for key in ("attention_rows", "gdn_rows", "attention_faults", "gdn_faults"):
        require(
            [row.get("row_id") for row in r30_local_replay.get(key, [])]
            == [row.get("row_id") for row in r30_oracle.get(key, [])],
            f"R30 local replay {key} row-order drift",
        )

    require(
        r29_observer.get("schema_version")
        == "forkaudit-r29-independent-gdn-observer-result-v1"
        and r29_observer.get("status") == "completed_valid_scientific_execution"
        and r29_observer.get("valid_cell_count") == 2
        and r29_observer.get("all_candidate_comparisons_exact") is True
        and r29_observer.get("all_independent_verdicts_passed") is True,
        "R29 source-distinct observer validity drift",
    )
    require(
        r29_observer.get("preregistration_sha256")
        == r29_expected_sha256[R29_OBSERVER_PREREGISTRATION]
        and r29_observer.get("source_ledger_raw_sha256")
        == r29_expected_sha256[R29_OBSERVER_SOURCE_LEDGER],
        "R29 source-distinct observer source binding drift",
    )
    observer_boundary = r29_observer.get("independence_boundary", {})
    require(
        observer_boundary
        == {
            "candidate_generated": [
                "model and cache objects",
                "request execution",
                "candidate qcomem-gdn-storage-witness-v1 rows",
            ],
            "cpu_replay_candidate_imports": False,
            "independent_observer_generated": [
                "tensor descriptors and content digests from live PyTorch tensors",
                "opaque stable object and storage tokens",
                "byte-overlap and exact-alias relations",
                "policy-specific phase verdicts",
                "persistent/incomplete/completed lifecycle verdicts",
                "candidate-row comparison",
            ],
            "observer_candidate_imports": False,
            "raw_addresses_serialized": False,
            "same_process": True,
            "setup_tensor_objects_pinned_against_allocator_aba": True,
            "shared_trusted_dependency": "PyTorch tensor/storage API",
        },
        "R29 source-distinct observer independence boundary drift",
    )
    require(
        r29_observer_replay.get("schema_version")
        == "forkaudit-r29-independent-gdn-observer-replay-v1"
        and r29_observer_replay.get("passed") is True
        and r29_observer_replay.get("valid_cell_count") == 2
        and r29_observer_replay.get("phase_count") == 6
        and r29_observer_replay.get("candidate_capture_replay_imported") is False
        and r29_observer_replay.get("candidate_passed_booleans_authoritative")
        is False,
        "R29 source-distinct observer replay boundary drift",
    )
    observer_phase_reports = [
        phase
        for cell in r29_observer_replay.get("cell_reports", [])
        for phase in cell.get("phase_reports", [])
    ]
    observer_lifecycle_reports = [
        cell.get("lifecycle", {})
        for cell in r29_observer_replay.get("cell_reports", [])
    ]
    observer_descriptor_comparisons = sum(
        int(phase.get("row_count", -1)) for phase in observer_phase_reports
    )
    observer_relation_coordinates = sum(
        int(phase.get("relation_count", -1)) for phase in observer_phase_reports
    )
    observer_exact_relations = sum(
        int(phase.get("exact_alias_comparisons", -1))
        for lifecycle in observer_lifecycle_reports
        for phase in lifecycle.get("phase_reports", [])
    )
    observer_descriptor_mismatches = sum(
        int(
            phase.get("independent_candidate_comparison", {}).get(
                "descriptor_mismatch_count", -1
            )
        )
        for cell in r29_observer.get("cells", [])
        for phase in cell.get("phases", [])
    )
    observer_relation_mismatches = sum(
        int(
            phase.get("independent_candidate_comparison", {}).get(
                "relation_mismatch_count", -1
            )
        )
        for cell in r29_observer.get("cells", [])
        for phase in cell.get("phases", [])
    )
    observer_serialized_rows = sum(
        len(phase.get(name, {}).get("rows", []))
        for cell in r29_observer.get("cells", [])
        for phase in cell.get("phases", [])
        for name in (
            "independent_before_candidate_capture",
            "independent_after_candidate_capture",
            "candidate_capture",
        )
    )
    require(
        observer_descriptor_comparisons
        == r29_observer_replay.get("independently_recomputed_row_observations")
        == 1_080
        and observer_relation_coordinates
        == r29_observer_replay.get("independently_recomputed_pair_relations")
        == 96_660
        and observer_exact_relations == 240
        and observer_relation_coordinates - observer_exact_relations == 96_420
        and observer_descriptor_mismatches == 0
        and observer_relation_mismatches == 0
        and len(observer_phase_reports) == 6
        and all(phase.get("candidate_comparison_exact") is True for phase in observer_phase_reports)
        and len(observer_lifecycle_reports) == 2
        and all(report.get("passed") is True for report in observer_lifecycle_reports)
        and observer_serialized_rows == 3_240,
        "R29 source-distinct observer numeric gate drift",
    )

    require(
        r29_two_stream.get("schema_version")
        == "qcomem-forkaudit-true-concurrent-lifecycle-result-v1"
        and r29_two_stream.get("status") == "completed"
        and r29_two_stream.get("scientific_execution_completed") is True
        and r29_two_stream.get("concurrency_treatment_valid") is True
        and r29_two_stream.get("scientific_run_valid") is True
        and r29_two_stream.get("formal_evidence_eligible") is True
        and r29_two_stream.get("primary_success") is True,
        "R29 two-stream formal validity drift",
    )
    require(
        r29_two_stream.get("design_preregistration_raw_sha256")
        == r29_expected_sha256[R29_TWO_STREAM_DESIGN],
        "R29 two-stream design binding drift",
    )
    require(
        r29_two_stream_replay.get("schema_version")
        == "qcomem-forkaudit-true-concurrent-lifecycle-replay-v1"
        and r29_two_stream_replay.get("status") == "completed"
        and r29_two_stream_replay.get("independent_replay_passed") is True
        and r29_two_stream_replay.get("scientific_run_valid") is True
        and r29_two_stream_replay.get("primary_success") is True,
        "R29 two-stream replay validity drift",
    )
    two_stream_overlaps = r29_two_stream_replay.get("concurrent", {}).get(
        "overlap_ms"
    )
    require(
        isinstance(two_stream_overlaps, list)
        and len(two_stream_overlaps) == 2
        and close(float(two_stream_overlaps[0]), 260.8020808696747)
        and close(float(two_stream_overlaps[1]), 95.43945229053497)
        and r29_two_stream_replay.get("serialized", {}).get("overlap_ms") == [],
        "R29 two-stream interval-overlap drift",
    )
    two_stream_oracle = r29_two_stream.get("output_oracle", {})
    two_stream_rows = two_stream_oracle.get("rows", [])
    two_stream_sidecar = r29_two_stream.get("sidecars", {}).get("serialized", {})
    two_stream_sidecar_records = two_stream_sidecar.get("records", [])
    two_stream_scalar_pairs = sum(
        int(record.get("shape", [0, 0])[-1]) for record in two_stream_sidecar_records
    )
    two_stream_unique_digests = {
        str(record.get("content_sha256")) for record in two_stream_sidecar_records
    }
    two_stream_kv_pairs = sum(
        len(row.get("layer_sha256", {}))
        for row in r29_two_stream.get("concurrent", {}).get("final_logical_kv", [])
    )
    two_stream_gdn = r29_two_stream.get("concurrent", {}).get(
        "final_gdn_state", []
    )
    require(
        two_stream_oracle.get("sample_count") == len(two_stream_rows) == 4
        and two_stream_oracle.get("all_full_vocab_logits_torch_equal") is True
        and two_stream_oracle.get("all_generated_tokens_equal") is True
        and close(float(two_stream_oracle.get("maximum_abs_error")), 0.0)
        and all(row.get("torch_equal") is True for row in two_stream_rows)
        and all(row.get("token_equal") is True for row in two_stream_rows)
        and two_stream_sidecar.get("record_count") == 4
        and two_stream_sidecar.get("terminal_exact_byte_coverage") is True
        and two_stream_scalar_pairs == 993_280
        and len(two_stream_unique_digests) == 3
        and two_stream_kv_pairs == 20
        and len(two_stream_gdn) == 2
        and sum(int(row.get("tensor_count", -1)) for row in two_stream_gdn)
        == 120,
        "R29 two-stream semantic/state denominator drift",
    )
    require(
        r29_two_stream.get("cross_arm")
        == {
            "full_vocab_logits_torch_equal": True,
            "generated_tokens_equal": True,
            "final_logical_kv_equal": True,
            "final_gdn_state_equal": True,
            "source_document_immutable_both_arms": True,
            "ownership_receipts_passed_both_arms": True,
            "lifecycle_replay_passed_both_arms": True,
            "concurrent_phases_with_positive_stream_overlap": 2,
        },
        "R29 two-stream cross-arm gate drift",
    )
    two_stream_not_established = [
        "simultaneous execution of individual CUDA kernels",
        "native vLLM-engine scheduling or continuous batching",
        "production server end-to-end correctness",
        "throughput, latency, QPS, capacity, or memory savings",
        "cross-model, cross-runtime, or cross-hardware generality",
        "safety for arbitrary cancellation timing or in-flight reclamation",
    ]
    require(
        r29_two_stream.get("claim_boundary", {}).get("not_established")
        == two_stream_not_established
        and r29_two_stream_replay.get("claim_boundary", {}).get("not_established")
        == two_stream_not_established,
        "R29 two-stream claim boundary drift",
    )

    require(
        r29_local_overhead.get("schema_version")
        == "forkaudit-r29-local-replay-overhead-result-v1"
        and r29_local_overhead.get("evidence_id") == "E-R29-LOCAL-REPLAY-OVERHEAD"
        and r29_local_overhead.get("scientific_run_valid") is True,
        "R29 local replay-overhead validity drift",
    )
    require(
        r29_local_overhead_validation.get("schema_version")
        == "forkaudit-r29-local-replay-overhead-validation-v1"
        and r29_local_overhead_validation.get("status")
        == "verified_complete_bounded_local_measurement"
        and r29_local_overhead_validation.get("scientific_run_valid") is True
        and r29_local_overhead_validation.get("preregistration_sha256")
        == r29_expected_sha256[R29_LOCAL_OVERHEAD_PREREGISTRATION]
        and r29_local_overhead_validation.get("result_sha256")
        == r29_expected_sha256[R29_LOCAL_OVERHEAD_RESULT]
        and all(
            value is True
            for value in r29_local_overhead_validation.get(
                "validity_gates", {}
            ).values()
        )
        and len(r29_local_overhead_validation.get("validity_gates", {})) == 5,
        "R29 local replay-overhead validation gate drift",
    )
    local_warmups = r29_local_overhead.get("warmup_rows", [])
    local_measured = r29_local_overhead.get("measured_rows", [])
    local_footprint_before = r29_local_overhead.get("footprint_before", {})
    local_footprint_after = r29_local_overhead.get("footprint_after", {})
    local_summary = r29_local_overhead.get("summary", {})
    local_wall_values = [float(row.get("wall_seconds")) for row in local_measured]
    require(
        len(local_warmups) == 1
        and len(local_measured) == 5
        and all(row.get("exit_code") == 0 for row in local_warmups + local_measured)
        and all(row.get("measured") is False for row in local_warmups)
        and all(row.get("measured") is True for row in local_measured)
        and len({row.get("stdout_sha256") for row in local_measured}) == 1
        and local_footprint_before
        == {
            "logical_bytes": 892_284_156,
            "logical_mib": 850.9484825134277,
            "regular_files": 630,
        }
        and local_footprint_after
        == {"logical_bytes": 892_284_156, "regular_files": 630}
        and local_summary.get("repetitions") == 5
        and close(
            float(local_summary.get("median_wall_seconds")),
            statistics.median(local_wall_values),
        )
        and close(float(local_summary.get("median_wall_seconds")), 102.706536375)
        and close(float(local_summary.get("min_wall_seconds")), 101.372219083)
        and close(float(local_summary.get("max_wall_seconds")), 103.061830875)
        and close(float(local_summary.get("median_child_user_seconds")), 98.323767)
        and close(float(local_summary.get("median_child_system_seconds")), 2.199897),
        "R29 local replay-overhead numeric gate drift",
    )
    require(
        r29_local_overhead.get("host")
        == {
            "cpu_brand": "Apple M4 Pro",
            "logical_cpu_count": 12,
            "machine": "arm64",
            "physical_memory_bytes": 25_769_803_776,
            "python": "3.9.6",
            "release": "25.4.0",
            "system": "Darwin",
        }
        and r29_local_overhead.get("claim_boundary")
        == (
            "Warm-cache CPU-only offline replay and logical package footprint "
            "only; no live-capture, production-latency, download, extraction, "
            "or engineering-effort claim."
        ),
        "R29 local replay-overhead host/boundary drift",
    )

    require(
        r29_live_overhead.get("schema_version")
        == "qcomem-forkaudit-r29-live-overhead-result-v1"
        and r29_live_overhead.get("status") == "completed"
        and r29_live_overhead.get("scientific_execution_completed") is True
        and r29_live_overhead.get("scientific_run_valid") is True
        and r29_live_overhead.get("formal_evidence_eligible") is True
        and r29_live_overhead.get("protocol")
        == "qcomem-forkaudit-paired-live-request-overhead-v1"
        and r29_live_overhead.get("design_preregistration_raw_sha256")
        == r29_expected_sha256[R29_LIVE_OVERHEAD_PREREGISTRATION],
        "R29 live-overhead formal validity/source binding drift",
    )
    require(
        r29_live_overhead_replay.get("schema_version")
        == "qcomem-forkaudit-r29-live-overhead-replay-v1"
        and r29_live_overhead_replay.get("replay_passed") is True
        and r29_live_overhead_replay.get("scientific_run_valid_recomputed") is True
        and r29_live_overhead_replay.get("formal_evidence_eligible_recomputed")
        is True
        and r29_live_overhead_replay.get("warmup_excluded_from_estimands") is True
        and r29_live_overhead_replay.get("negative_numeric_deltas_preserved")
        is True,
        "R29 live-overhead replay-v2 validity drift",
    )
    live_validity = r29_live_overhead.get("validity", {})
    require(
        live_validity
        == {
            "warmup_pair_count": 1,
            "warmup_discarded_from_estimands": True,
            "measured_pair_count": 5,
            "alternating_schedule_verified": True,
            "all_pair_semantic_oracles_exact": True,
            "all_live_receipts_valid": True,
            "negative_numeric_deltas_removed": False,
        },
        "R29 live-overhead validity receipt drift",
    )
    expected_live_rows = [
        {
            "pair_index": 0,
            "execution_order": ["baseline", "instrumented"],
            "baseline_slot": 0,
            "instrumented_slot": 1,
            "baseline_wall_time_ns": 153_620_082,
            "instrumented_wall_time_ns": 695_913_362,
            "paired_wall_delta_ns": 542_293_280,
            "paired_wall_ratio": 4.530093676164031,
            "baseline_incremental_peak_allocated_bytes": 83_841_024,
            "instrumented_incremental_peak_allocated_bytes": 82_397_184,
            "paired_incremental_peak_delta_bytes": -1_443_840,
            "instrumented_audit_artifact_bytes": 3_063_111,
            "pair_valid": True,
        },
        {
            "pair_index": 1,
            "execution_order": ["instrumented", "baseline"],
            "baseline_slot": 1,
            "instrumented_slot": 0,
            "baseline_wall_time_ns": 149_398_099,
            "instrumented_wall_time_ns": 665_831_432,
            "paired_wall_delta_ns": 516_433_333,
            "paired_wall_ratio": 4.456759734272121,
            "baseline_incremental_peak_allocated_bytes": 82_397_184,
            "instrumented_incremental_peak_allocated_bytes": 83_841_024,
            "paired_incremental_peak_delta_bytes": 1_443_840,
            "instrumented_audit_artifact_bytes": 3_063_111,
            "pair_valid": True,
        },
        {
            "pair_index": 2,
            "execution_order": ["baseline", "instrumented"],
            "baseline_slot": 1,
            "instrumented_slot": 0,
            "baseline_wall_time_ns": 157_067_992,
            "instrumented_wall_time_ns": 665_128_436,
            "paired_wall_delta_ns": 508_060_444,
            "paired_wall_ratio": 4.234652952079505,
            "baseline_incremental_peak_allocated_bytes": 83_841_024,
            "instrumented_incremental_peak_allocated_bytes": 82_397_184,
            "paired_incremental_peak_delta_bytes": -1_443_840,
            "instrumented_audit_artifact_bytes": 3_063_111,
            "pair_valid": True,
        },
        {
            "pair_index": 3,
            "execution_order": ["instrumented", "baseline"],
            "baseline_slot": 0,
            "instrumented_slot": 1,
            "baseline_wall_time_ns": 153_939_241,
            "instrumented_wall_time_ns": 652_493_805,
            "paired_wall_delta_ns": 498_554_564,
            "paired_wall_ratio": 4.238645070362534,
            "baseline_incremental_peak_allocated_bytes": 82_397_184,
            "instrumented_incremental_peak_allocated_bytes": 83_841_024,
            "paired_incremental_peak_delta_bytes": 1_443_840,
            "instrumented_audit_artifact_bytes": 3_063_111,
            "pair_valid": True,
        },
        {
            "pair_index": 4,
            "execution_order": ["baseline", "instrumented"],
            "baseline_slot": 0,
            "instrumented_slot": 1,
            "baseline_wall_time_ns": 154_218_228,
            "instrumented_wall_time_ns": 666_378_600,
            "paired_wall_delta_ns": 512_160_372,
            "paired_wall_ratio": 4.321010613609177,
            "baseline_incremental_peak_allocated_bytes": 83_841_024,
            "instrumented_incremental_peak_allocated_bytes": 82_397_184,
            "paired_incremental_peak_delta_bytes": -1_443_840,
            "instrumented_audit_artifact_bytes": 3_063_111,
            "pair_valid": True,
        },
    ]
    live_summary = r29_live_overhead.get("paired_summary", {})
    live_replayed_summary = r29_live_overhead_replay.get(
        "paired_summary_recomputed", {}
    )
    require(
        live_summary.get("rows") == expected_live_rows
        and live_replayed_summary.get("rows") == expected_live_rows,
        "R29 live-overhead five-pair row drift",
    )
    expected_live_summary_scalars = {
        "measured_pair_count": 5,
        "warmup_pairs_included": 0,
        "median_paired_wall_delta_ns": 512_160_372,
        "min_paired_wall_delta_ns": 498_554_564,
        "max_paired_wall_delta_ns": 542_293_280,
        "median_paired_wall_ratio": 4.321010613609177,
        "median_paired_incremental_peak_delta_bytes": -1_443_840,
        "median_instrumented_audit_artifact_bytes": 3_063_111,
        "negative_numeric_deltas_preserved": True,
        "statistical_significance_claimed": False,
    }
    require_exact_fields(
        live_summary, expected_live_summary_scalars, "R29 live-overhead formal summary"
    )
    require_exact_fields(
        live_replayed_summary,
        expected_live_summary_scalars,
        "R29 live-overhead replayed summary",
    )
    live_pairs = [
        r29_live_overhead.get("warmup_pair", {}),
        *r29_live_overhead.get("measured_pairs", []),
    ]
    require(
        len(live_pairs) == 6
        and live_pairs[0].get("warmup") is True
        and live_pairs[0].get("discarded_from_estimands") is True
        and all(pair.get("pair_valid") is True for pair in live_pairs)
        and all(pair.get("source_document_immutable") is True for pair in live_pairs)
        and all(pair.get("persistent_gdn_immutable") is True for pair in live_pairs)
        and all(
            pair.get("semantic_oracle", {}).get("full_vocab_logits_torch_equal")
            is True
            and pair.get("semantic_oracle", {}).get("generated_token_equal") is True
            and pair.get("semantic_oracle", {}).get("baseline_token_id") == 353
            and pair.get("semantic_oracle", {}).get("instrumented_token_id") == 353
            and close(
                float(pair.get("semantic_oracle", {}).get("max_abs_error")), 0.0
            )
            for pair in live_pairs
        )
        and all(
            pair.get("cells", {}).get("baseline", {}).get("capture_policy")
            == "optional-forkaudit-capture-disabled"
            and pair.get("cells", {}).get("baseline", {}).get(
                "ownership_receipt_enabled"
            )
            is False
            and pair.get("cells", {}).get("baseline", {}).get(
                "audit_artifact_bytes"
            )
            == 0
            and pair.get("cells", {}).get("instrumented", {}).get(
                "capture_policy"
            )
            == "full-live-capture-and-ownership-receipt"
            and pair.get("cells", {}).get("instrumented", {}).get(
                "ownership_receipt_enabled"
            )
            is True
            for pair in live_pairs
        ),
        "R29 live-overhead pair semantic/receipt gate drift",
    )
    live_sidecar = r29_live_overhead.get("semantic_sidecar", {})
    live_sidecar_rows = live_sidecar.get("records", [])
    require(
        live_sidecar.get("schema_version")
        == "qcomem-forkaudit-r29-live-overhead-logits-v1"
        and live_sidecar.get("sha256")
        == r29_expected_sha256[R29_LIVE_OVERHEAD_SEMANTIC_SIDECAR]
        and live_sidecar.get("bytes") == 11_919_360
        and live_sidecar.get("record_count") == len(live_sidecar_rows) == 12
        and live_sidecar.get("terminal_exact_byte_coverage") is True
        and all(row.get("shape") == [1, 248_320] for row in live_sidecar_rows)
        and all(row.get("nbytes") == 993_280 for row in live_sidecar_rows)
        and all(row.get("token_id") == 353 for row in live_sidecar_rows)
        and len({row.get("content_sha256") for row in live_sidecar_rows}) == 1,
        "R29 live-overhead semantic-sidecar gate drift",
    )
    live_replay_semantic_rows = r29_live_overhead_replay.get("semantic_rows", [])
    live_replay_artifact_rows = r29_live_overhead_replay.get("artifact_rows", [])
    require(
        len(live_replay_semantic_rows) == 6
        and all(
            row.get("full_vocab_logits_exact") is True
            and row.get("generated_token_equal") is True
            and row.get("baseline_token_id") == 353
            and row.get("instrumented_token_id") == 353
            for row in live_replay_semantic_rows
        )
        and len(live_replay_artifact_rows) == 6
        and all(row.get("passed") is True for row in live_replay_artifact_rows)
        and all(
            row.get("capture_bytes") == 2_950_400
            and row.get("tensor_record_count") == 50
            and row.get("request_gdn_rebound_tensor_count") == 60
            for row in live_replay_artifact_rows
        )
        and all(
            row.get("receipt_bytes") == 112_711
            and row.get("artifact_bytes") == 3_063_111
            for row in live_replay_artifact_rows[1:]
        ),
        "R29 live-overhead replay semantic/artifact gate drift",
    )
    require(
        r29_live_overhead.get("claim_boundary")
        == {
            "established_only_if_valid": (
                "paired request-step overhead of this exact live ForkAudit "
                "capture/ownership/receipt configuration on one frozen "
                "Qwen3.5-35B-A3B vLLM-Q16 H20 stack and one frozen PG19/query input"
            ),
            "not_established": [
                "model loading or document-prefill overhead",
                "throughput, QPS, serving capacity, or concurrency scaling",
                "native vLLM-engine continuous batching or production server latency",
                "cross-model, cross-runtime, cross-hardware, or cross-input generality",
                "zero-cost auditing",
                "statistical significance from five pairs",
            ],
        },
        "R29 live-overhead claim boundary drift",
    )
    require_exact_fields(
        integrated.get("live_audit_overhead", {}),
        {
            "status": (
                "verified_bounded_formal_with_postexecution_replay_only_correction"
            ),
            "artifact_root": "evidence/r29_live_overhead/formal_run_20260825b",
            "preregistration_sha256": r29_expected_sha256[
                R29_LIVE_OVERHEAD_PREREGISTRATION
            ],
            "pre_second_execution_amendment_sha256": r29_expected_sha256[
                R29_LIVE_OVERHEAD_PRESECOND_AMENDMENT
            ],
            "postexecution_replay_only_amendment_sha256": r29_expected_sha256[
                R29_LIVE_OVERHEAD_REPLAY_AMENDMENT
            ],
            "formal_result_sha256": r29_expected_sha256[R29_LIVE_OVERHEAD_RESULT],
            "independent_replay_v2_sha256": r29_expected_sha256[
                R29_LIVE_OVERHEAD_REPLAY
            ],
            "terminal_raw_and_replay_v2_ledger_sha256": r29_expected_sha256[
                R29_LIVE_OVERHEAD_TERMINAL_LEDGER
            ],
            "discarded_warmup_pairs": 1,
            "measured_pairs": 5,
            "valid_measured_pairs": 5,
            "median_paired_wall_delta_ns": 512_160_372,
            "minimum_paired_wall_delta_ns": 498_554_564,
            "maximum_paired_wall_delta_ns": 542_293_280,
            "median_paired_wall_ratio": 4.321010613609177,
            "median_instrumented_audit_artifact_bytes": 3_063_111,
            "median_paired_incremental_peak_delta_bytes": -1_443_840,
            "exact_full_vocabulary_logit_pairs_including_warmup": 6,
            "exact_generated_token_pairs_including_warmup": 6,
            "instrumented_tensor_records_per_pair": 50,
            "request_gdn_rebound_rows_per_pair": 60,
            "statistical_significance_claimed": False,
            "boundary": (
                "One input, one request step, one H20, one discarded warmup and five "
                "measured audit-off/on pairs. The complete synchronous "
                "capture/receipt path is compared with the same ledger/kernel path "
                "with optional auditing disabled. This is not production serving "
                "latency, GPU-kernel-only overhead, throughput, QPS, capacity, "
                "cross-input generality, or a memory-saving claim. The replay-v2 "
                "correction changed only the offline verifier for the raw request-GDN "
                "witness schema; it preserved the already-read formal result and all "
                "raw scientific artifacts and performed no GPU rerun."
            ),
        },
        "R29 integrated live-audit overhead",
    )

    require(
        r29_heldout_author_freeze.get("schema_version")
        == "forkaudit-r29-heldout-fault-author-freeze-v1"
        and r29_heldout_author_freeze.get("status")
        == "frozen_before_cross_execution"
        and r29_heldout_author_freeze.get("fault_author_ran_candidate_cases")
        is False
        and r29_heldout_author_freeze.get("candidate_outputs_exist_at_freeze")
        is False
        and r29_heldout_author_freeze.get("suite_raw_sha256")
        == r29_expected_sha256[R29_HELDOUT_SUITE]
        and r29_heldout_author_freeze.get("suite_canonical_json_sha256")
        == "1913b62010d5020b18c56170569131928e580fb27879fd7c62618cbfe3966867"
        and r29_heldout_author_freeze.get("fault_author_code_ledger_raw_sha256")
        == r29_expected_sha256[R29_HELDOUT_AUTHOR_LEDGER],
        "R29 held-out preexecution author freeze drift",
    )
    require(
        r29_heldout_readiness.get("schema_version")
        == "forkaudit-r29-heldout-cross-execution-readiness-v3"
        and r29_heldout_readiness.get("status")
        == "v3_frozen_for_parent_audit_before_h20_execution"
        and r29_heldout_readiness.get("third_attempt_candidate_outputs_exist")
        is False
        and r29_heldout_readiness.get(
            "scientific_design_input_or_estimator_changed_by_v3"
        )
        is False
        and r29_heldout_readiness.get("gpu_execution_authorized_by_this_file")
        is False,
        "R29 held-out preexecution readiness drift",
    )
    heldout_integrated = integrated.get("heldout_realistic_pattern_faults", {})
    require(
        heldout_integrated
        == {
            "status": "pending_no_scientific_result",
            "scientific_result_available": False,
            "claim_authorized": False,
            "result": None,
            "boundary": (
                "Frozen protocol and attempted execution metadata only. No terminal "
                "scientifically eligible aggregate exists, so no per-fault outcome, "
                "rate, or manuscript result is authorized."
            ),
        },
        "R29 held-out pending/no-claim boundary drift",
    )
    require_exact_fields(
        integrated.get("source_distinct_gdn_observer", {}),
        {
            "status": "verified_internal_superseded_by_r33_out_of_process_capture",
            "active_manuscript_support": False,
            "superseded_by_evidence_id": "E-R33-OUT-OF-PROCESS-GDN-CAPTURE-A",
            "artifact_root": "evidence/r29_independent_observer/formal_run_20260825a",
            "preregistration_sha256": r29_expected_sha256[
                R29_OBSERVER_PREREGISTRATION
            ],
            "source_ledger_sha256": r29_expected_sha256[R29_OBSERVER_SOURCE_LEDGER],
            "preexecution_amendment_sha256": r29_expected_sha256[
                R29_OBSERVER_AMENDMENT
            ],
            "formal_result_sha256": r29_expected_sha256[R29_OBSERVER_RESULT],
            "independent_replay_sha256": r29_expected_sha256[R29_OBSERVER_REPLAY],
            "terminal_ledger_sha256": r29_expected_sha256[
                R29_OBSERVER_TERMINAL_LEDGER
            ],
            "fresh_n2_cells": 2,
            "phase_points": 6,
            "candidate_descriptor_comparisons": 1_080,
            "paired_relation_coordinates": 96_660,
            "exact_relation_coordinates": 240,
            "disjoint_relation_coordinates": 96_420,
            "descriptor_mismatches": 0,
            "relation_mismatches": 0,
            "phase_verdicts_passed": 6,
            "lifecycle_verdicts_passed": 2,
            "serialized_rows": 3_240,
            "boundary": (
                "Internal superseded corroboration only. The source-distinct "
                "observer and candidate inspect the same candidate-created objects, "
                "receive the same phase/completion labels, and trust the same "
                "PyTorch storage API. R29 is not active manuscript support for "
                "capture independence; it is not independent producer recapture, "
                "external ground truth, KV observation, binary attestation, or a "
                "guarantee against transient writes restored between snapshots."
            ),
        },
        "R29 integrated source-distinct observer",
    )
    require_exact_fields(
        integrated.get("bounded_two_stream_lifecycle", {}),
        {
            "status": "verified_bounded_two_stream_interval_overlap",
            "artifact_root": "evidence/r29_true_concurrency/formal_run_20260825b",
            "design_sha256": r29_expected_sha256[R29_TWO_STREAM_DESIGN],
            "pre_second_execution_amendment_sha256": r29_expected_sha256[
                R29_TWO_STREAM_AMENDMENT
            ],
            "source_ledger_sha256": r29_expected_sha256[
                R29_TWO_STREAM_SOURCE_LEDGER
            ],
            "formal_result_sha256": r29_expected_sha256[R29_TWO_STREAM_RESULT],
            "independent_replay_sha256": r29_expected_sha256[R29_TWO_STREAM_REPLAY],
            "terminal_ledger_sha256": r29_expected_sha256[
                R29_TWO_STREAM_TERMINAL_LEDGER
            ],
            "pre_cancel_full_call_interval_overlap_ms": 260.8020808696747,
            "post_reclaim_full_call_interval_overlap_ms": 95.43945229053497,
            "host_threads": 2,
            "distinct_cuda_streams": 2,
            "lifecycle_indexed_full_logit_samples": 4,
            "unique_logit_digests": 3,
            "cross_arm_fp32_scalar_pairs": 993_280,
            "maximum_fp32_error": 0.0,
            "exact_tokens": 4,
            "terminal_logical_kv_layer_pairs": 20,
            "gdn_slot_aggregates": 2,
            "gdn_tensors_summarized": 120,
            "boundary": (
                "CUDA-event-bracketed full-model-call intervals include queueing, "
                "resource waits, and host-enqueue gaps. They establish two distinct "
                "streams simultaneously in flight, not simultaneous individual GPU "
                "kernels, speedup, native continuous batching, production scheduling, "
                "or arbitrary/in-flight cancellation. Cancellation and reclamation "
                "occur only after synchronization."
            ),
        },
        "R29 integrated bounded two-stream lifecycle",
    )
    require_exact_fields(
        integrated.get("local_replay_overhead", {}),
        {
            "status": "verified_complete_bounded_local_measurement",
            "artifact_root": "evidence/r29_overhead",
            "preregistration_sha256": r29_expected_sha256[
                R29_LOCAL_OVERHEAD_PREREGISTRATION
            ],
            "result_sha256": r29_expected_sha256[R29_LOCAL_OVERHEAD_RESULT],
            "validation_sha256": r29_expected_sha256[
                R29_LOCAL_OVERHEAD_VALIDATION
            ],
            "terminal_ledger_sha256": r29_expected_sha256[
                R29_LOCAL_OVERHEAD_TERMINAL_LEDGER
            ],
            "discarded_warmups": 1,
            "measured_replays": 5,
            "successful_replays": 5,
            "package_regular_files": 630,
            "package_logical_bytes": 892_284_156,
            "package_logical_mib": 850.9484825134277,
            "median_wall_seconds": 102.706536375,
            "minimum_wall_seconds": 101.372219083,
            "maximum_wall_seconds": 103.061830875,
            "median_user_seconds": 98.323767,
            "median_system_seconds": 2.199897,
            "boundary": (
                "Already-unpacked warm-cache CPU-only complete replay on one Apple "
                "M4 Pro with 12 logical CPUs, 24 GiB, Darwin 25.4, and Python 3.9.6. "
                "It excludes producer capture, H20 live audit, cold "
                "download/extraction, device memory, service latency, and "
                "integration effort."
            ),
        },
        "R29 integrated local replay overhead",
    )
    r28_expected_counts = {
        "cases": 18,
        "completed_semantic_paths": 5,
        "token_only_catches_among_completed": 0,
        "full_logit_catches_among_completed": 1,
        "other_forkaudit_gate_catches": 2,
        "production_assertion_catches": 1,
        "fault_payload_aborts": 1,
        "measured_non_forkaudit_escapes": 4,
        "operational_invalid": 0,
    }
    r28_assurance = assurance["target_gate_suppression_matrix"]
    r28_integrated = integrated["target_gate_suppression_matrix"]
    for source_name, matrix in (
        ("assurance boundary", r28_assurance),
        ("integrated results", r28_integrated),
    ):
        observed_counts = {
            key: matrix.get(key) for key in r28_expected_counts
        }
        require(
            observed_counts == r28_expected_counts,
            f"R28 target-gate-suppression counts drift in {source_name}: "
            f"{observed_counts}",
        )
    require(
        r28_assurance["matched_clean_controls_passed"] == 9
        and r28_assurance["target_suppressed_mutants"] == 9,
        "R28 assurance clean/mutant lane counts drift",
    )
    require(
        r28_integrated["matched_clean_cases"] == 9
        and r28_integrated["target_suppressed_mutant_cases"] == 9,
        "R28 integrated clean/mutant lane counts drift",
    )
    require(
        r28_assurance["interpretation"]
        == (
            "prospective per-fault same-system comparison; the injected M8 "
            "sentinel is not a production detector and no row is pooled into a rate"
        ),
        "R28 assurance interpretation boundary drift",
    )
    require(
        r28_integrated["scope"]
        == (
            "M1--M9, each with a fresh matched-clean case and a case suppressing "
            "only its named target gate; eight distinct H20s on the fixed Qwen3.5 stack"
        ),
        "R28 separate-case scope drift",
    )
    require(
        r28_integrated["boundary"]
        == (
            "Prospective per-fault same-system comparison on nine designed faults, "
            "not blind/naturally occurring faults, a population detection or "
            "false-positive rate, another model/runtime, independent producer "
            "recapture, or general superiority over conventional testing. N/O is "
            "missingness after an earlier classified stop; M8's injected sentinel "
            "is not a production detector."
        ),
        "R28 per-fault/non-rate boundary drift",
    )
    r28_expected_per_fault = {
        "M1": "completes; exact token and FP32 logits unchanged",
        "M2": "completes; exact token and FP32 logits unchanged",
        "M3": "other ForkAudit gate KV_ACTIVE_BLOCK_OWNERSHIP",
        "M4": "other ForkAudit gate gdn_completed_vs_peers_disjoint",
        "M5": (
            "completes; token unchanged; FP32 logits change with max_abs=0.75 "
            "and relative_l2=0.031414290548977195"
        ),
        "M6": "completes; exact token and FP32 logits unchanged",
        "M7": "completes; exact token and FP32 logits unchanged",
        "M8": "exact injected payload sentinel; not a production detector",
        "M9": "exact allowlisted paired-view production assertion",
    }
    require(
        r28_integrated["per_fault"] == r28_expected_per_fault,
        "R28 per-fault outcome map drift",
    )
    require(
        r28_integrated["status"]
        == "verified_postexecution_field_source_correction_byte_replayed"
        and r28_integrated["corrected_replay_byte_identical"] is True
        and r28_integrated["candidate_outputs_rewritten"] is False,
        "R28 replay/candidate-byte boundary drift",
    )
    first_gate = assurance["primary_first_gate_localization"]
    require(
        first_gate["matched_clean_controls_passed"]
        == first_gate["mutants_rejected_at_predeclared_first_gate"]
        == 9,
        "first-gate positive-control coverage drift",
    )
    require(
        first_gate["post_injection_output_or_semantic_digest_available"] == 0,
        "post-injection semantic availability drift",
    )
    require(
        first_gate["lifecycle_storage_fault_ids"] == ["M1", "M3", "M4", "M5"]
        and first_gate["binding_call_fault_ids"] == ["M2", "M6", "M7", "M8", "M9"],
        "first-gate receipt-family partition drift",
    )
    scheduler_extension = assurance["scheduler_extension"]
    require(
        scheduler_extension["clean_rank_geometry_cells_passed"] == 16
        and scheduler_extension["preregistered_fault_trials_at_expected_gate"] == 48
        and scheduler_extension["expected_gate_misses"] == 0,
        "scheduler-extension coverage drift",
    )
    footprint = assurance["artifact_footprint"]
    require(
        footprint["source_complete_files"] == 628
        and footprint["source_complete_bytes"] == 892_144_066
        and footprint["raw_bound_artifacts"] == 536
        and footprint["raw_bound_bytes"] == 888_785_811,
        "artifact-footprint drift",
    )
    require(
        footprint["capture_time_measured"] is False
        and footprint["complete_replay_time_measured"] is False,
        "unregistered timing appeared in artifact footprint",
    )
    require(serving["schema"] == "forkaudit-related-serving-panel-v1",
            "serving-panel schema drift")
    require(serving["status"] == "verified_complete", "serving-panel status drift")
    require(serving["fixed_protocol"]["workloads"] == 8, "serving workload count drift")
    serving_rows = {(row["system"], row["phase"]): row for row in serving["rows"]}
    require(len(serving_rows) == 7, "serving-panel row coverage drift")
    for system in ("vLLM 0.26", "SGLang 0.5.17"):
        off = serving_rows[(system, "cache_off")]
        on = serving_rows[(system, "cache_on")]
        require(close(float(off["mean_f1_points"]), float(on["mean_f1_points"])),
                f"{system} cache-off/on F1 drift")
        require(on["cache_hits"] == on["prediction_exact_vs_off"] == "8/8",
                f"{system} cache-hit/exactness drift")
    require(close(serving_rows[("vLLM 0.26", "cache_off")]["median_ttft_seconds"], 1.4450775813311338),
            "vLLM cache-off TTFT drift")
    require(close(serving_rows[("vLLM 0.26", "cache_on")]["median_ttft_seconds"], 0.17894425056874752),
            "vLLM cache-on TTFT drift")
    require(close(serving_rows[("SGLang 0.5.17", "cache_off")]["median_ttft_seconds"], 0.2833681385964155),
            "SGLang cache-off TTFT drift")
    require(close(serving_rows[("SGLang 0.5.17", "cache_on")]["median_ttft_seconds"], 0.1464698724448681),
            "SGLang cache-on TTFT drift")
    require(hypic["scientific_run_valid"] is True, "HYPIC scientific validity drift")
    require(hypic["protocol_validity"] == "passed", "HYPIC protocol validity drift")
    require(hypic["official_commit"] == "98147c01909004e66d98bcb18b886927d41b0ee5",
            "HYPIC official source drift")
    require(HYPIC_SUMMARY.read_bytes() == HYPIC_INDEPENDENT.read_bytes(),
            "HYPIC independent aggregate differs")
    require(
        sha256(HYPIC_STORE_ACCEPTANCE) == EXPECTED_HYPIC_STORE_ACCEPTANCE_SHA256,
        "HYPIC Store acceptance drift",
    )
    require(
        hypic_store.get("schema_version")
        == "hypic-rwd5-trial1892234-external-store-acceptance-v1",
        "HYPIC Store acceptance schema drift",
    )
    require(
        hypic_store.get("status") == "passed_external_replay_16_of_16",
        "HYPIC Store acceptance status drift",
    )
    require(hypic_store.get("job_id") == 247699, "HYPIC Store job drift")
    require(hypic_store.get("trial_id") == 1892234, "HYPIC Store trial drift")
    require(
        hypic_store.get("official_commit")
        == "98147c01909004e66d98bcb18b886927d41b0ee5",
        "HYPIC Store source drift",
    )
    require(
        hypic_store.get("terminal_cells", {}).get("passed")
        == hypic_store.get("terminal_cells", {}).get("expected")
        == 16,
        "HYPIC Store cell coverage drift",
    )
    require(
        hypic_store.get("external_replay", {}).get("passed") is True
        and hypic_store.get("external_replay", {}).get("rows") == 16,
        "HYPIC Store external replay drift",
    )
    hypic_store_medians = {
        "prefix_cache": 146_309_120,
        "transition_rope_recompute": 339_834_880,
    }
    for mode, expected_median in hypic_store_medians.items():
        mode_store = hypic_store.get("modes", {}).get(mode, {})
        payload_bytes = mode_store.get("payload_bytes")
        require(
            isinstance(payload_bytes, list)
            and len(payload_bytes) == 8
            and all(isinstance(value, int) and value > 0 for value in payload_bytes),
            f"HYPIC Store row coverage drift: {mode}",
        )
        require(
            int(statistics.median(payload_bytes)) == expected_median
            and mode_store.get("median_payload_bytes") == expected_median,
            f"HYPIC Store median drift: {mode}",
        )
    hypic_expected = {
        "full_recompute": (39.13722478238607, 0.21664894558489323, "--", "reference"),
        "prefix_cache": (39.13722478238607, 0.0721846129745245, "8/8", "8/8"),
        "transition_rope_recompute": (39.63325652841781, 0.10121727548539639, "8/8", "7/8"),
    }
    for mode, (expected_f1, expected_ttft, expected_hits, expected_exact) in hypic_expected.items():
        row = serving_rows[("HYPIC/SGLang 0.5.14", mode)]
        require(close(float(row["mean_f1_points"]), expected_f1), f"HYPIC {mode} F1 drift")
        require(close(float(row["median_ttft_seconds"]), expected_ttft), f"HYPIC {mode} TTFT drift")
        require(row["cache_hits"] == expected_hits, f"HYPIC {mode} cache-hit drift")
        require(row["prediction_exact_vs_off"] == expected_exact,
                f"HYPIC {mode} prediction-exact drift")
    require(
        gdn_oracle["status"] == "verified_bounded_fully_preregistered",
        "GDN oracle status drift",
    )
    require(gdn_oracle["post_execution_amendment_used"] is False, "GDN amendment drift")
    gdn_results = gdn_oracle["results"]
    require(gdn_results["all_clean_rows_pass"] is True, "GDN oracle clean row failed")
    require(
        gdn_results["all_seeded_wrong_transitions_rejected"] is True,
        "GDN seeded wrong transition escaped",
    )
    require(gdn_results["clean_rows_passed"] == gdn_results["clean_rows_total"] == 4,
            "GDN clean row count drift")
    require(
        gdn_results["seeded_wrong_transitions_rejected"]
        == gdn_results["seeded_wrong_transitions_total"]
        == 4,
        "GDN fault row count drift",
    )
    gdn_max_output_relative_l2 = float(gdn_results["maximum_clean_output_relative_l2"])
    gdn_max_state_relative_l2 = float(gdn_results["maximum_clean_state_relative_l2"])
    require(
        close(gdn_max_output_relative_l2, 0.0016522899472646122),
        "GDN maximum output relative-L2 drift",
    )
    require(
        close(gdn_max_state_relative_l2, 1.3907660746265477e-7),
        "GDN maximum state relative-L2 drift",
    )

    a4 = json.loads(A4_AGGREGATE.read_text())
    rr2_derived = json.loads(RR2_DERIVED.read_text())
    require(rr2_derived["factorial_adjacent_cross_n_exact"] is True,
            "RR2 adjacent cross-N exactness drift")
    require(rr2_derived["factorial_adjacent_cross_n_comparison_count"] == 288,
            "RR2 adjacent cross-N count drift")
    require(
        rr2_derived["invariants"]["adjacent_cross_n_comparison_count_is_288"] is True,
        "RR2 adjacent cross-N invariant drift",
    )
    require(a4["scientific_run_valid"] is True, "A4 scientific validity drift")
    require(
        a4["scientific_outcome"] == "valid_negative_transformers_runtime_transfer",
        "A4 outcome drift",
    )
    require(a4["passed"] is False, "A4 negative unexpectedly marked passed")
    require(a4["rank_count"] == a4["distinct_gpu_uuids"] == 8, "A4 rank/GPU drift")
    require(a4["distinct_pg19_train_books"] == 8, "A4 book count drift")
    require(
        a4["status_vector"]
        == ["full", "full", "full", "not_applicable", "partial", "full", "full"],
        "A4 target status drift",
    )
    require(
        a4["fault_outcome_counts"]
        == {
            "clean_false_positive": 16,
            "detected_expected_predicate": 24,
            "detected_wrong_predicate": 0,
            "escaped": 0,
        },
        "A4 fault outcome drift",
    )
    a4_ledger = json.loads(A4_ARTIFACT_LEDGER.read_text())
    for row in a4_ledger["rows"]:
        path = A4_ROOT / row["path"]
        require(path.is_file(), f"A4 artifact missing: {row['path']}")
        require(path.stat().st_size == row["bytes"], f"A4 artifact size drift: {row['path']}")
        require(sha256(path) == row["sha256"], f"A4 artifact SHA drift: {row['path']}")
    rows = summary["capacity_matrix"]
    require(summary["passed"] is True, "primary aggregate is not passed")
    require(summary["resident_counts"] == [1, 2, 4, 8, 16, 32], "N grid drift")
    require(len(rows) == 6, "capacity row count drift")
    require(summary["rank_count"] == summary["world_size"] == 8, "rank count drift")
    require(summary["same_kernel_full_logit_token_logical_kv_gdn_exact_fraction"] == 1.0,
            "primary exactness fraction drift")
    require(summary["cross_n_prefix_isolation_exact"] is True, "cross-N gate drift")
    require(summary["pg19_train_only"] is True, "PG-19 train-only gate drift")
    require(not summary["longbench_consumed"], "LongBench unexpectedly consumed")
    require(not summary["test_v2_consumed"], "test-v2 unexpectedly consumed")

    n32 = next(row for row in rows if row["resident_count"] == 32)
    summary_rows_by_n = {int(row["resident_count"]): row for row in rows}
    fresh_pool = (
        n32["fresh"]["source_document_allocated_nbytes"]
        + n32["fresh"]["source_private_reservation_nbytes"]
        + n32["fresh"]["fresh_duplicate_document_allocated_nbytes"]
        + n32["fresh"]["fresh_duplicate_private_reservation_nbytes"]
    )
    reuse_pool = (
        n32["reuse"]["source_document_allocated_nbytes"]
        + n32["reuse"]["source_private_reservation_nbytes"]
    )
    require(fresh_pool == 2960 * 2**20, "fresh N=32 pool drift")
    require(reuse_pool == 240 * 2**20, "reuse N=32 pool drift")
    counterfactual_fresh_pool = 85 * 32 * 2**20
    counterfactual_difference = counterfactual_fresh_pool - reuse_pool
    require(counterfactual_difference == 2480 * 2**20, "counterfactual N=32 difference drift")

    shard_paths = sorted(SHARDS.glob("multifork-resident-shard-*.json"))
    require(len(shard_paths) == 8, "raw shard count drift")
    paired_requests = paired_generation_steps = logical_kv_pairs = gdn_pairs = fused_calls = 0
    unique_query_hashes: set[str] = set()
    phase_windows: dict[int, dict[str, dict[str, list[int]]]] = {
        n: {
            arm: {"post_pack_production": [], "full_recorded_lifecycle": []}
            for arm in ("fresh", "reuse")
        }
        for n in summary["resident_counts"]
    }
    for path in shard_paths:
        shard = json.loads(path.read_text())
        query_rows = shard["query_bank"]["rows"]
        require(len(query_rows) == 32, f"query bank size drift: {path.name}")
        unique_query_hashes.update(row["query_token_ids_sha256"] for row in query_rows)
        for row in shard["rows"]:
            n = row["resident_count"]
            require(n in phase_windows, f"unexpected resident count in {path.name}: {n}")
            for arm in ("fresh", "reuse"):
                arm_row = row[arm]
                setup_peak = arm_row["resident_setup"]["allocator_after"]["peak_allocated_bytes"]
                generation_peak = arm_row["generation"][
                    "production_allocator_before_exactness"
                ]["peak_allocated_bytes"]
                post_pack_peak = max(setup_peak, generation_peak)
                require(
                    post_pack_peak
                    == arm_row["generation_only"]["production_absolute_peak_allocated_bytes"],
                    f"{path.name} N={n} {arm} generation-only production peak drift",
                )
                require(
                    post_pack_peak
                    == arm_row["setup_plus_generation"]["combined_absolute_peak_allocated_bytes"],
                    f"{path.name} N={n} {arm} legacy combined peak drift",
                )
                lifecycle_peak = max(
                    arm_row["common_document_prefill"]["allocator_after"]["peak_allocated_bytes"],
                    arm_row["common_q16_pack"]["allocator_after"]["peak_allocated_bytes"],
                    setup_peak,
                    generation_peak,
                )
                phase_windows[n][arm]["post_pack_production"].append(post_pack_peak)
                phase_windows[n][arm]["full_recorded_lifecycle"].append(lifecycle_peak)
            paired_requests += n
            paired_generation_steps += n * 8
            logical_kv_pairs += n * 10
            gdn_pairs += n
            for arm in (row["fresh"], row["reuse"]):
                fused_calls += sum(item["total_calls"] for item in arm["intercepts"])
    require(len(unique_query_hashes) == 256, "unique query count drift")
    require(paired_requests == 504, "paired request count drift")
    require(paired_generation_steps == 4032, "paired generation-step count drift")
    arm_specific_model_steps = 2 * paired_generation_steps
    require(arm_specific_model_steps == 8064, "arm-specific model-step count drift")
    require(logical_kv_pairs == 5040, "logical-KV pair count drift")
    require(gdn_pairs == 504, "GDN pair count drift")
    require(fused_calls == 80640, "fused-call count drift")
    allocator_windows_by_n: list[dict[str, int | bool]] = []
    for n in summary["resident_counts"]:
        replayed: dict[str, dict[str, int]] = {}
        for arm in ("fresh", "reuse"):
            replayed[arm] = {}
            for window in ("post_pack_production", "full_recorded_lifecycle"):
                values = phase_windows[n][arm][window]
                require(len(values) == 8, f"N={n} {arm} {window} rank coverage drift")
                require(len(set(values)) == 1, f"N={n} {arm} {window} differs across ranks")
                replayed[arm][window] = int(statistics.median(values))
        summary_row = summary_rows_by_n[n]
        require(
            replayed["fresh"]["post_pack_production"]
            == summary_row["fresh"]["production_absolute_peak_allocated_median_bytes"],
            f"N={n} fresh summary production peak drift",
        )
        require(
            replayed["reuse"]["post_pack_production"]
            == summary_row["reuse"]["production_absolute_peak_allocated_median_bytes"],
            f"N={n} reuse summary production peak drift",
        )
        allocator_windows_by_n.append(
            {
                "resident_count": n,
                "fresh_post_pack_production_peak_bytes": replayed["fresh"]["post_pack_production"],
                "reuse_post_pack_production_peak_bytes": replayed["reuse"]["post_pack_production"],
                "fresh_full_recorded_lifecycle_peak_bytes": replayed["fresh"]["full_recorded_lifecycle"],
                "reuse_full_recorded_lifecycle_peak_bytes": replayed["reuse"]["full_recorded_lifecycle"],
                "rank_values_identical": True,
            }
        )

    n32_windows = next(row for row in allocator_windows_by_n if row["resident_count"] == 32)
    fresh_production_peak = int(n32_windows["fresh_post_pack_production_peak_bytes"])
    reuse_production_peak = int(n32_windows["reuse_post_pack_production_peak_bytes"])
    production_peak_difference = fresh_production_peak - reuse_production_peak
    production_peak_fraction = production_peak_difference / fresh_production_peak
    fresh_lifecycle_peak = int(n32_windows["fresh_full_recorded_lifecycle_peak_bytes"])
    reuse_lifecycle_peak = int(n32_windows["reuse_full_recorded_lifecycle_peak_bytes"])
    lifecycle_peak_difference = fresh_lifecycle_peak - reuse_lifecycle_peak
    lifecycle_peak_fraction = lifecycle_peak_difference / fresh_lifecycle_peak
    require(production_peak_difference == 2_857_268_224, "N=32 production allocator difference drift")
    require(close(production_peak_fraction, 0.038289283508797234), "N=32 production allocator fraction drift")
    require(fresh_lifecycle_peak == 74_623_183_360, "fresh N=32 lifecycle peak drift")
    require(reuse_lifecycle_peak == 72_407_176_192, "reuse N=32 lifecycle peak drift")
    require(lifecycle_peak_difference == 2_216_007_168, "N=32 lifecycle difference drift")
    require(close(lifecycle_peak_fraction, 0.029695961338307614), "N=32 lifecycle fraction drift")

    required_fragments = [
        "96 KV-by-GDN ownership configurations",
        "288 adjacent-fan-out comparisons",
        "each completed request's 60 mutable GDN tensors as disjoint from bases and peers",
        (
            "At $N=32$, switching full-copy to shared-document KV reduces the final "
            "allocated delta by $2.672\\gib$ with materialized GDN setup and "
            "$2.661\\gib$ with borrowed setup."
        ),
        "it does not measure continuous batching",
        "Full-prefix KV cache & Q16 & 140.34",
        "CoMem state & Q8 & 15.89",
        "13.38 & 39.14",
        "7.15 & 39.14",
        "\\input{tables/h20_deployment_table.tex}",
        "\\input{tables/first_gate_localization_table.tex}",
        "\\input{tables/related_serving_table.tex}",
        "\\input{tables/related_work_reported_context.tex}",
        "vLLM and SGLang each record 8/8 prefix hits",
        "preserve 8/8 cache-off predictions",
        "not a cross-framework leaderboard",
        "official-code HYPIC",
        "HYPIC gives 39.63, 0.101 s, and 17.67",
        "published Qwen3.5 panels use TP=2",
        "24 timing/quality cells",
        "16 retained-state cells",
        "Prefix Cache retains a median 139.53 MiB",
        "HYPIC retains 324.09 MiB",
        "8.78\\times",
        "20.39\\times",
        "4,404-entry terminal artifact ledger",
        "candidate-import-free NumPy recurrence",
        "0.001652",
        "1.39\\times10^{-7}",
        "post-native-q/k-normalization boundary",
        "not a detection rate",
        "post-injection token, logit, logical-KV, or GDN digest",
        "N/O",
        "all 16 clean rank--geometry cells",
        "all 48 preregistered schedule-fault trials",
        "Full synchronous capture and persistence are costly and intended for offline debugging or CI.",
        "288 adjacent-fan-out",
    ]
    r28_required_fragments = [
        "All nine preregistered primary faults reach their frozen first gates",
        (
            "when each named gate is separately suppressed, five reach semantics, "
            "where tokens catch $0/5$ and exact logits catch $1/5$."
        ),
        (
            "For the five semantic-complete designed faults, token equality catches "
            "$0/5$ and exact logits catch $1/5$"
        ),
        "In a separate preregistered 18-case matrix, only that gate is suppressed.",
        "Unallowlisted exceptions are invalid; the sentinel is not a production detector.",
        "All nine controls pass and no case is invalid.",
        (
            "Five suppressed mutants complete: tokens catch none, exact logits catch "
            "M5, and M1/M2/M6/M7 retain both."
        ),
        (
            "M3/M4 reach redundant audit gates, M9 a production assertion, and M8 "
            "its injected sentinel."
        ),
        (
            "For the five semantic-complete designed faults, token equality catches "
            "$0/5$ and exact logits catch $1/5$, whereas the named ForkAudit gates "
            "reject all five in their separate all-gates-on executions."
        ),
        (
            "The target-gate-suppression package rechecks 18 cases and 24 FP32 "
            "sidecars"
        ),
        (
            "does not estimate a population rate, held-out-fault coverage, or "
            "false-positive rate."
        ),
    ]
    required_fragments.extend(r28_required_fragments)
    r29_required_fragments = [
        (
            "Two host threads on distinct CUDA streams produce positive full-model-call "
            "interval overlap both before cancellation"
        ),
        (
            "All four tokens and full-logit samples, 20 terminal KV layer pairs, "
            "two 60-tensor GDN aggregates, and ownership/lifecycle receipts match "
            "the serialized reference."
        ),
        (
            "This establishes call-interval overlap and one clean quiescent transition, "
            "not simultaneous kernels or arbitrary in-flight cancellation."
        ),
        "Cancellation occurs only after synchronization, so the cohort does not test arbitrary or in-flight reclamation.",
        "there is no per-kernel overlap trace.",
    ]
    required_fragments.extend(r29_required_fragments)
    r33_required_fragments = [
        (
            "The receiver processes derive 1,080 storage descriptors and all 96,660 "
            "pair relations; frozen replay passes 6/6 phase and 2/2 lifecycle verdicts."
        ),
        "Both observer processes differ from the producer process",
        "receives no candidate rows, verdicts, or live phase/policy/completion labels.",
        "the producer still enumerates and semantically binds the frozen slots.",
        (
            "Every matched clean case passes complete coverage, byte binding, "
            "full-horizon execution, and exact allocator restoration; every mutant "
            "also completes and is rejected first by its pre-frozen primary predicate."
        ),
        "Of the four pairs with comparable call cardinality, three preserve every FP32-logit sidecar byte",
        "These are five fixed frozen outcomes, not an estimate over a fault population.",
        (
            "The designer--executor-separated package binds the PDF-only author "
            "freeze, a lifecycle-amended executor, five clean/mutant pairs, detached "
            "clean and pair replays, and 208 terminal files."
        ),
        "Passing all five does not estimate unseen-fault recall, false-positive rate",
        "\\input{tables/r33_fresh_heldout_table.tex}",
    ]
    required_fragments.extend(r33_required_fragments)
    r35_required_fragments = [
        "A retrospective reproduction of one historical integration defect",
        "three archived-coordinate and five additional frozen-input cells",
        "the pre-fix path reaches the expected authenticated binding predicate",
        "token, complete FP32 logits, terminal request GDN, and logical KV are exact",
        "A persistent-base content invariant also catches 8/8 historical cells",
        "The repaired path privatizes the 30 convolution states; it is storage-clean and exact",
        "one previously encountered integration defect, not a natural-bug corpus",
        "earlier owner/layer/family-specific localization",
        "\\input{tables/r35_historical_alias_table.tex}",
    ]
    required_fragments.extend(r35_required_fragments)
    r30_required_fragments = [
        (
            "Separately, 44 captured-boundary numerical rows pass and 44 "
            "wrong-operator controls fail."
        ),
        "20 attention rows spanning all ten full-attention layers (160 query positions)",
        "24 GDN rows spanning 12 of 30 recurrent layers (192 token transitions)",
        "maximum attention relative-$L_2$ is $0.001897$",
        "maximum GDN output/state relative-$L_2$ is $0.002073$/$2.08\\times10^{-7}$",
        "All 20 attention and 24 GDN seeded wrong-operator controls are rejected.",
        "candidate-import-free CPU/NumPy FP32 replay",
        "The expanded numerical-sweep package binds 272 sidecars totaling 140,199,936 bytes",
        "No package independently re-executes the model, enumerates OS/driver allocations, or attests compiled dispatch",
        "not upstream-activation, independent-capture, full-model, or end-to-end validation",
        "39 state-appended input tokens (32 query + 7 generated-token feedback)",
        "Trace coverage and replay verdict",
        "pass at Python scope",
    ]
    required_fragments.extend(r30_required_fragments)
    missing = [
        fragment for fragment in required_fragments
        if fragment not in normalized_manuscript
    ]
    require(not missing, f"manuscript is missing required fragments: {missing}")
    ambiguous_fragments = [
        "allocator deltas and absolute peaks",
        "Peak materialize",
        "Peak shared",
    ]
    ambiguous = [
        fragment for fragment in ambiguous_fragments if fragment in visible_manuscript
    ]
    require(not ambiguous, f"manuscript contains ambiguous peak wording: {ambiguous}")
    forbidden_visible_fragments = [
        "seven of eight prediction texts",
        "changes one of eight prediction texts",
        "7/8 texts",
        r"\texttt{n/r}",
        "HYPIC n/r",
        "CoMem state (caveated)",
        "Q16 is shown only",
        "Prediction exact vs.",
        "4.321",
        "512.160",
        "850.95",
        "102.706",
        "892,284,156",
        "892,144,066",
        "888,785,811",
        "847.61",
        "tab:live-overhead",
        "1.0398",
        "1.0306",
        "12.33",
        "Single-request fair control",
        "1.0090",
        "a separately implemented same-process observer matches all 1,080 candidate row descriptors",
        "2,160 observer before/after rows plus 1,080 candidate rows",
    ]
    forbidden_visible = [
        fragment
        for fragment in forbidden_visible_fragments
        if fragment in normalized_manuscript
    ]
    require(
        not forbidden_visible,
        f"manuscript exposes internal-only negative/missing-result fields: {forbidden_visible}",
    )
    require(
        re.search(r"\bRC\b", visible_manuscript) is None
        and "receipt-complete" not in normalized_manuscript.lower(),
        "manuscript retains deprecated RC/receipt-complete terminology",
    )
    require(
        "runtime-independent" not in normalized_manuscript.lower()
        and "runtime independent" not in normalized_manuscript.lower(),
        "manuscript makes a runtime-independence claim",
    )
    r28_forbidden_visible_fragments = [
        "incremental detection rate",
        "unique-catch rate",
        "overall detection rate",
        "pooled detection rate",
        "we report a detection rate",
        "we estimate a detection rate",
        "population detection rate of",
        "false-positive rate of",
        "false-negative rate of",
        "the injected sentinel is a production detector",
        "M8 production detector",
        "M8 production assertion",
        "nine of nine faults detected",
        "9/9 faults detected",
        "demonstrates detector completeness",
        "establishes detector completeness",
    ]
    normalized_manuscript_lower = normalized_manuscript.lower()
    r28_forbidden_visible = [
        fragment
        for fragment in r28_forbidden_visible_fragments
        if fragment.lower() in normalized_manuscript_lower
    ]
    require(
        not r28_forbidden_visible,
        "manuscript violates the R28 separate-execution/per-fault/non-rate "
        f"boundary: {r28_forbidden_visible}",
    )
    anonymous_summary = json.loads(ANONYMOUS_SUMMARY.read_text())
    anonymous_shard_paths = sorted(
        ANONYMOUS_SHARDS.glob("multifork-resident-shard-*.json")
    )
    require(len(anonymous_shard_paths) == 8, "anonymous raw shard count drift")
    require(
        anonymous_summary["resident_counts"] == summary["resident_counts"],
        "anonymous resident-count grid drift",
    )
    require(
        anonymous_summary["capacity_matrix"] == summary["capacity_matrix"],
        "anonymous capacity matrix differs from original numeric evidence",
    )
    require(
        anonymous_summary["same_kernel_full_logit_token_logical_kv_gdn_exact_fraction"]
        == summary["same_kernel_full_logit_token_logical_kv_gdn_exact_fraction"],
        "anonymous exactness result drift",
    )
    metrics = json.loads(ARTIFACT_METRICS.read_text())
    require(metrics["evidence_namespace"] == "anonymous_derivative", "artifact namespace drift")
    require(
        metrics["source"]["summary_sha256"] == sha256(ANONYMOUS_SUMMARY),
        "artifact metrics do not bind the distributed anonymous summary",
    )
    metric_capacity_by_n = {
        int(row["resident_count"]): row for row in metrics["capacity"]
    }
    require(set(metric_capacity_by_n) == set(summary["resident_counts"]), "metric N grid drift")
    for replayed in allocator_windows_by_n:
        n = int(replayed["resident_count"])
        metric_row = metric_capacity_by_n[n]
        for key in (
            "fresh_post_pack_production_peak_allocated_bytes",
            "reuse_post_pack_production_peak_allocated_bytes",
            "fresh_full_recorded_lifecycle_peak_allocated_bytes",
            "reuse_full_recorded_lifecycle_peak_allocated_bytes",
        ):
            replay_key = key.replace("_allocated", "")
            require(
                int(metric_row[key]) == int(replayed[replay_key]),
                f"N={n} metric {key} disagrees with raw phase replay",
            )

    audit = {
        "status": "passed",
        "primary_evidence_namespace": "anonymous_derivative",
        "evidence_namespaces": {
            "original_run": {
                "summary_sha256": sha256(SUMMARY),
                "shard_sha256": [sha256(path) for path in shard_paths],
            },
            "anonymous_derivative": {
                "summary_sha256": sha256(ANONYMOUS_SUMMARY),
                "shard_sha256": [sha256(path) for path in anonymous_shard_paths],
                "manifest_sha256": sha256(ANONYMOUS_MANIFEST),
            },
        },
        "manuscript_sha256": sha256(MANUSCRIPT),
        "raw_shards": len(shard_paths),
        "unique_query_hashes": len(unique_query_hashes),
        "paired_request_instances": paired_requests,
        "paired_full_vocab_step_comparisons": paired_generation_steps,
        "arm_specific_model_steps": arm_specific_model_steps,
        "earlier_capacity_cross_n_coverage": {
            "eligible_unique_queries": 128,
            "singleton_unique_queries": 128,
            "larger_fanout_comparisons": 248,
        },
        "primary_factorial_adjacent_cross_n": {
            "derived_summary_sha256": sha256(RR2_DERIVED),
            "comparison_count": 288,
            "exact": True,
            "edges": ["N=1->8", "N=8->32"],
            "rank_count": 8,
            "arm_count": 4,
        },
        "logical_kv_layer_pairs": logical_kv_pairs,
        "gdn_state_pairs": gdn_pairs,
        "fused_attention_calls": fused_calls,
        "gdn_oracle": {
            "validation_sha256": sha256(GDN_ORACLE_VALIDATION),
            "classification": gdn_oracle["status"],
            "preregistration_sha256": gdn_oracle["final_preregistration_raw_sha256"],
            "clean_rows": gdn_results["clean_rows_total"],
            "seeded_faults": gdn_results["seeded_wrong_transitions_total"],
            "max_output_relative_l2": gdn_max_output_relative_l2,
            "max_state_relative_l2": gdn_max_state_relative_l2,
        },
        "r30_expanded_captured_boundary_oracle": {
            "status": r30_validation["status"],
            "preregistration_sha256": sha256(R30_PREREGISTRATION),
            "capture_manifest_sha256": sha256(R30_CAPTURE_MANIFEST),
            "canonical_oracle_result_sha256": sha256(R30_ORACLE_RESULT),
            "validation_report_sha256": sha256(R30_VALIDATION_REPORT),
            "raw_artifacts_ledger_sha256": sha256(R30_RAW_ARTIFACTS_LEDGER),
            "terminal_products_ledger_sha256": sha256(
                R30_TERMINAL_PRODUCTS_LEDGER
            ),
            "candidate_execution_attempts": 1,
            "input_cases": r30_coverage["input_cases"],
            "attention_clean_rows": r30_results[
                "attention_clean_rows_total"
            ],
            "attention_layers_covered": r30_results[
                "attention_full_layers_covered"
            ],
            "attention_layers_total": r30_results[
                "attention_full_layers_total"
            ],
            "attention_query_positions": r30_results[
                "attention_query_positions"
            ],
            "maximum_clean_attention_relative_l2": r30_results[
                "maximum_clean_attention_relative_l2"
            ],
            "gdn_clean_rows": r30_results["gdn_clean_rows_total"],
            "gdn_layers_covered": r30_results["gdn_layers_covered"],
            "gdn_layers_total": r30_results["gdn_layers_total"],
            "gdn_token_transitions": r30_results["gdn_token_transitions"],
            "maximum_clean_gdn_output_relative_l2": r30_results[
                "maximum_clean_gdn_output_relative_l2"
            ],
            "maximum_clean_gdn_state_relative_l2": r30_results[
                "maximum_clean_gdn_state_relative_l2"
            ],
            "attention_seeded_faults_rejected": r30_results[
                "attention_seeded_faults_rejected"
            ],
            "gdn_seeded_faults_rejected": r30_results[
                "gdn_seeded_faults_rejected"
            ],
            "local_sidecar_files": len(r30_receipts),
            "local_sidecar_bytes": r30_sidecar_bytes,
            "local_payload_bytes": r30_payload_bytes,
            "local_candidate_import_free_replay_passed": True,
            "independent_capture": False,
            "end_to_end_validation": False,
        },
        "serving_panel": {
            "summary_sha256": sha256(SERVING_PANEL),
            "systems": ["vLLM 0.26", "SGLang 0.5.17", "HYPIC/SGLang 0.5.14"],
            "vllm_sglang_raw_shards_per_system": 16,
            "hypic_formal_cells": 24,
            "vllm_sglang_cache_hits_per_system": "8/8",
            "vllm_sglang_cache_off_on_predictions_exact_per_system": "8/8",
            "hypic_summary_sha256": sha256(HYPIC_SUMMARY),
            "hypic_independent_aggregate_byte_identical": True,
            "hypic_prefix_prediction_exact": "8/8",
            "hypic_prediction_exact": "7/8",
        },
        "hypic_store": {
            "acceptance_sha256": sha256(HYPIC_STORE_ACCEPTANCE),
            "status": hypic_store["status"],
            "job_id": hypic_store["job_id"],
            "trial_id": hypic_store["trial_id"],
            "accepted_cells": hypic_store["terminal_cells"]["passed"],
            "external_replay_rows": hypic_store["external_replay"]["rows"],
            "prefix_cache_median_payload_bytes": hypic_store_medians["prefix_cache"],
            "hypic_median_payload_bytes": hypic_store_medians["transition_rope_recompute"],
        },
        "transformers_runtime_transfer": {
            "aggregate_sha256": sha256(A4_AGGREGATE),
            "artifact_ledger_sha256": sha256(A4_ARTIFACT_LEDGER),
            "artifact_rows_verified": len(a4_ledger["rows"]),
            "scientific_outcome": a4["scientific_outcome"],
            "status_vector": a4["status_vector"],
            "fault_outcome_counts": a4["fault_outcome_counts"],
        },
        "target_gate_suppression_matrix": {
            "assurance_boundary_sha256": sha256(ASSURANCE_BOUNDARY),
            "integrated_results_sha256": sha256(INTEGRATED_RESULTS),
            "counts": r28_expected_counts,
            "matched_clean_cases": 9,
            "target_suppressed_mutant_cases": 9,
            "per_fault_outcomes": r28_expected_per_fault,
            "separate_execution_not_pooled_as_rate": True,
            "m8_sentinel_is_production_detector": False,
        },
        "r29_supporting_evidence": {
            "source_distinct_gdn_observer": {
                "formal_result_sha256": sha256(R29_OBSERVER_RESULT),
                "independent_replay_sha256": sha256(R29_OBSERVER_REPLAY),
                "terminal_ledger_sha256": sha256(R29_OBSERVER_TERMINAL_LEDGER),
                "fresh_n2_cells": 2,
                "phase_points": len(observer_phase_reports),
                "candidate_descriptor_comparisons": observer_descriptor_comparisons,
                "paired_relation_coordinates": observer_relation_coordinates,
                "exact_relation_coordinates": observer_exact_relations,
                "disjoint_relation_coordinates": (
                    observer_relation_coordinates - observer_exact_relations
                ),
                "descriptor_mismatches": observer_descriptor_mismatches,
                "relation_mismatches": observer_relation_mismatches,
                "serialized_rows": observer_serialized_rows,
                "same_process": True,
                "independent_producer_recapture": False,
            },
            "bounded_two_stream_lifecycle": {
                "formal_result_sha256": sha256(R29_TWO_STREAM_RESULT),
                "independent_replay_sha256": sha256(R29_TWO_STREAM_REPLAY),
                "terminal_ledger_sha256": sha256(R29_TWO_STREAM_TERMINAL_LEDGER),
                "call_interval_overlap_ms": two_stream_overlaps,
                "full_logit_samples": len(two_stream_rows),
                "cross_arm_fp32_scalar_pairs": two_stream_scalar_pairs,
                "unique_logit_digests": len(two_stream_unique_digests),
                "maximum_fp32_error": two_stream_oracle["maximum_abs_error"],
                "terminal_logical_kv_layer_pairs": two_stream_kv_pairs,
                "gdn_slot_aggregates": len(two_stream_gdn),
                "gdn_tensors_summarized": sum(
                    int(row["tensor_count"]) for row in two_stream_gdn
                ),
                "simultaneous_individual_kernels_established": False,
                "inflight_cancellation_safety_established": False,
            },
            "live_audit_overhead": {
                "formal_result_sha256": sha256(R29_LIVE_OVERHEAD_RESULT),
                "independent_replay_v2_sha256": sha256(R29_LIVE_OVERHEAD_REPLAY),
                "terminal_ledger_sha256": sha256(
                    R29_LIVE_OVERHEAD_TERMINAL_LEDGER
                ),
                **expected_live_summary_scalars,
                "exact_logit_and_token_pairs_including_warmup": len(
                    live_replay_semantic_rows
                ),
                "instrumented_tensor_records_per_pair": 50,
                "request_gdn_rebound_rows_per_pair": 60,
                "gpu_rerun_for_replay_v2": False,
                "memory_saving_claimed": False,
            },
            "local_complete_replay_overhead": {
                "result_sha256": sha256(R29_LOCAL_OVERHEAD_RESULT),
                "validation_sha256": sha256(R29_LOCAL_OVERHEAD_VALIDATION),
                "terminal_ledger_sha256": sha256(
                    R29_LOCAL_OVERHEAD_TERMINAL_LEDGER
                ),
                "discarded_warmups": len(local_warmups),
                "successful_measured_replays": len(local_measured),
                "package_regular_files": local_footprint_before["regular_files"],
                "package_logical_bytes": local_footprint_before["logical_bytes"],
                "median_wall_seconds": local_summary["median_wall_seconds"],
                "minimum_wall_seconds": local_summary["min_wall_seconds"],
                "maximum_wall_seconds": local_summary["max_wall_seconds"],
            },
            "heldout_realistic_pattern_faults": {
                "suite_raw_sha256": sha256(R29_HELDOUT_SUITE),
                "author_freeze_sha256": sha256(R29_HELDOUT_AUTHOR_FREEZE),
                "execution_input_v3_sha256": sha256(
                    R29_HELDOUT_EXECUTION_INPUT_V3
                ),
                "readiness_v3_sha256": sha256(R29_HELDOUT_READINESS_V3),
                "status": heldout_integrated["status"],
                "scientific_result_available": False,
                "claim_authorized": False,
                "result": None,
            },
        },
        "r33_supporting_evidence": {
            "out_of_process_gdn_capture": r33_capture_audit,
            "pdf_only_fresh_heldout_faults": r33_heldout_audit,
        },
        "r35_historical_alias_regression": r35_historical_alias_audit,
        "allocator_windows_by_n": allocator_windows_by_n,
        "n32": {
            "fresh_pool_mib": fresh_pool / 2**20,
            "reuse_pool_mib": reuse_pool / 2**20,
            "pool_ratio": fresh_pool / reuse_pool,
            "fresh_post_pack_production_peak_bytes": fresh_production_peak,
            "reuse_post_pack_production_peak_bytes": reuse_production_peak,
            "post_pack_production_peak_difference_bytes": production_peak_difference,
            "post_pack_production_peak_difference_gib": production_peak_difference / 2**30,
            "post_pack_production_peak_difference_fraction": production_peak_fraction,
            "fresh_full_recorded_lifecycle_peak_bytes": fresh_lifecycle_peak,
            "reuse_full_recorded_lifecycle_peak_bytes": reuse_lifecycle_peak,
            "full_recorded_lifecycle_peak_difference_bytes": lifecycle_peak_difference,
            "full_recorded_lifecycle_peak_difference_gib": lifecycle_peak_difference / 2**30,
            "full_recorded_lifecycle_peak_difference_fraction": lifecycle_peak_fraction,
            "source_free_counterfactual_fresh_pool_mib": counterfactual_fresh_pool / 2**20,
            "source_free_counterfactual_difference_mib": counterfactual_difference / 2**20,
        },
        "scope_checks": {
            "pg19_train_only": summary["pg19_train_only"],
            "longbench_consumed": summary["longbench_consumed"],
            "test_v2_consumed": summary["test_v2_consumed"],
            "timing_not_aggregated": summary[
                "timing_is_raw_validation_instrumented_single_observation_not_aggregated"
            ],
            "allocator_not_nvml": summary[
                "allocator_absolute_values_are_pytorch_allocator_not_nvml_or_total_model_capacity"
            ],
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
