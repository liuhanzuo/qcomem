from __future__ import annotations

"""Fail-closed local verifier for the frozen R35 H20 evidence bundle.

The verifier does not import candidate or runtime modules.  It checks the
frozen inputs, byte-for-byte output archive extraction, the complete
launch/aggregate/replay/raw/sidecar hash graph, and then invokes the replay and
aggregator programs from a freshly extracted frozen execution package.  The
deterministic result contains no temporary or host-absolute paths.
"""

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Mapping, Sequence


SCHEMA = "forkaudit-r35-local-evidence-verification-v1"
RUN_ID = "R35-HISTORICAL-ALIAS-20260826A"
ARCHIVE_ROOT_NAME = "r35-historical-alias-20260826a"
LANES = (
    "historical_pre_fix",
    "repaired_borrowed",
    "materialized_control",
)
RECEIPT_ORDER = (
    "frozen_input_and_request_provenance",
    "live_kv_ownership_and_construction_binding",
    "gdn_phase_storage_snapshot_and_pointer_free_replay",
    "advertised_scheduler_action_sequence_replay",
    "persistent_kv_and_gdn_immutability",
    "fresh_case_disposal_pending",
)
GATE = "gdn_completed_binding_rebound"
PAIRS = (
    "historical_pre_fix_vs_materialized_control",
    "historical_pre_fix_vs_repaired_borrowed",
    "repaired_borrowed_vs_materialized_control",
)
METRICS = (
    "greedy_token_exact",
    "full_fp32_logits_exact",
    "request0_terminal_gdn_content_exact",
    "logical_kv_content_exact",
    "persistent_base_content_only_invariant",
)
OUTCOMES = (
    "historical_expected_authenticated_first_gate_reproduced",
    "historical_output_only_exact_to_materialized",
    "repaired_storage_contract_clean",
    "materialized_storage_contract_clean",
    "repaired_semantic_and_terminal_exact_to_materialized",
)
HEADLINES = (
    "all_eight_historical_expected_gates_reproduced",
    "all_eight_historical_output_only_pairs_exact",
    "all_eight_repaired_storage_contracts_clean",
    "all_eight_materialized_storage_contracts_clean",
    "all_eight_repaired_pairs_semantic_and_terminal_exact",
    "registered_positive_headline_rule_satisfied",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_HASHES = {
    "preregistration.json": "6cffd4775a0969007b504adc4a091b8587f17d2f5317734e683fba242e598d74",
    "static-execution-input.json": "26be505aad84adbab18b3752928e31f391912661f116c4fe7a9c8a4f7dd338fa",
    "source.sha256": "672d847665519976a7ddf3d88ff969570723b7159b43cf02b8b6037862659051",
    "r35-execution-package.tar.gz": "cc979b249c6b6e214d377d377cb5d249e1441f78e469b179515e2ed0cfb174cf",
    "preexecution-resource-amendment.json": "a48557e64994c6a70615f8851e2c9b2e19ba44da2c7417fb331a860e3e58aad1",
    "preexecution-freeze-receipt.json": "e0bc2c335b197ca3f204aa039b4d218308bb5fa3e55a176d9a24d920d22e8c2f",
    "r35-historical-alias-20260826a-output.tar.gz": "93e177c7cb483aa3c1f02ec7e602f8af2fbc33fefe8deee45189af0963e4317d",
}

EXPECTED_FAULT_ISOLATION = {
    "r29_heldout_fault_suite_import_blocked": True,
    "r29_heldout_fault_suite_in_sys_modules": False,
    "generic_mutant_definition_module_passively_loaded": True,
    "mutation_requested": False,
    "mutation_applied": False,
}
EXPECTED_LANE_MUTATION = {
    "r29_heldout_fault_module_loaded": False,
    "generic_mutant_definition_module_passively_loaded": True,
    "mutation_requested": False,
    "mutation_applied": False,
    "mutation_event_count": 0,
}


class VerificationError(RuntimeError):
    """A missing artifact, binding drift, or failed independent replay."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing regular file: {path.name}")
    with path.open("rb") as handle:
        return sha256_stream(handle)


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def check_sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} SHA-256")
    return value


def exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    require(set(value) == keys, f"{label} schema drift")


def relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise VerificationError(f"artifact outside paper root: {path.name}") from exc


def safe_archive_name(name: str, label: str) -> PurePosixPath:
    path = PurePosixPath(name)
    require(
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and path.as_posix() == name,
        f"unsafe {label} member",
    )
    return path


def expected_run_files() -> set[str]:
    paths = {"aggregate.json", "launch-completion.json", "logs/aggregate.log"}
    paths.update(f"logs/rank-{rank}.log" for rank in range(8))
    paths.update(f"logs/replay-rank-{rank}.log" for rank in range(8))
    paths.update(f"replay/rank-{rank}-replay.json" for rank in range(8))
    for rank in range(8):
        paths.add(f"rank-{rank}/raw/rank-result.json")
        paths.update(
            f"rank-{rank}/raw/{lane}-full-fp32-logits.bin" for lane in LANES
        )
    return paths


def expected_run_directories() -> set[str]:
    paths = {"logs", "replay"}
    paths.update(f"rank-{rank}" for rank in range(8))
    paths.update(f"rank-{rank}/raw" for rank in range(8))
    return paths


def filesystem_inventory(root: Path) -> tuple[set[str], set[str]]:
    require(root.is_dir() and not root.is_symlink(), "missing extracted run root")
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "symlink in extracted output")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            require(relative not in files, "duplicate extracted file")
            files.add(relative)
        elif path.is_dir():
            require(relative not in directories, "duplicate extracted directory")
            directories.add(relative)
        else:
            raise VerificationError("non-file object in extracted output")
    return files, directories


def verify_output_archive(archive: Path, run_root: Path) -> dict[str, Any]:
    archive_sha = sha256_file(archive)
    require(
        archive_sha == EXPECTED_HASHES[archive.name],
        "output archive raw hash drift",
    )
    expected_files = expected_run_files()
    expected_directories = expected_run_directories()
    observed_files, observed_directories = filesystem_inventory(run_root)
    require(observed_files == expected_files, "extracted output file-set drift")
    require(observed_directories == expected_directories, "extracted output directory-set drift")

    archive_files: set[str] = set()
    archive_directories: set[str] = set()
    names: set[str] = set()
    with tarfile.open(archive, mode="r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            name = safe_archive_name(member.name, "output archive")
            require(member.name not in names, "duplicate output archive member")
            names.add(member.name)
            require(
                member.isfile() or member.isdir(),
                "link or special object in output archive",
            )
            require(name.parts[0] == ARCHIVE_ROOT_NAME, "output archive root drift")
            relative_parts = name.parts[1:]
            if not relative_parts:
                require(member.isdir(), "output archive root is not a directory")
                continue
            relative = PurePosixPath(*relative_parts).as_posix()
            if member.isdir():
                archive_directories.add(relative)
                continue
            archive_files.add(relative)
            target = run_root / Path(*relative_parts)
            require(target.is_file() and not target.is_symlink(), "archive payload target missing")
            extracted = handle.extractfile(member)
            require(extracted is not None, "output archive payload missing")
            with extracted:
                member_sha = sha256_stream(extracted)
            require(member.size == target.stat().st_size, "archive/extracted size drift")
            require(member_sha == sha256_file(target), "archive/extracted hash drift")
    require(archive_files == expected_files, "output archive file-set drift")
    require(archive_directories == expected_directories, "output archive directory-set drift")
    require(len(names) == 78, "output archive member cardinality")
    return {
        "path": archive.name,
        "sha256": archive_sha,
        "nbytes": archive.stat().st_size,
        "archive_member_count": 78,
        "archive_file_count": 59,
        "archive_directory_count": 19,
        "extracted_file_count": 59,
        "extracted_directory_count": 18,
        "archive_payloads_byte_exact_to_extracted": True,
    }


def parse_source_ledger(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise VerificationError(f"invalid source ledger: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        require(bool(raw.strip()), f"blank source-ledger row {line_number}")
        pieces = raw.split(None, 1)
        require(len(pieces) == 2, f"source-ledger row {line_number}")
        digest, raw_name = pieces
        check_sha(digest, f"source-ledger row {line_number}")
        name = raw_name.lstrip("*")
        safe_archive_name(name, "source-ledger")
        require(name not in rows, "duplicate source-ledger path")
        rows[name] = digest
    require(len(rows) == 11, "source-ledger row cardinality")
    return rows


def package_expected_paths(ledger: Mapping[str, str]) -> set[str]:
    base = set(ledger)
    base.update(
        {
            "paper_autonomous_multifork_iteration/evidence/r35_historical_alias_regression/preregistration.json",
            "paper_autonomous_multifork_iteration/evidence/r35_historical_alias_regression/static-execution-input.json",
            "paper_autonomous_multifork_iteration/evidence/r35_historical_alias_regression/source.sha256",
        }
    )
    metadata = {
        str(PurePosixPath(name).parent / ("._" + PurePosixPath(name).name))
        for name in base
    }
    return base | metadata


def extract_frozen_package(
    package: Path,
    destination: Path,
    expected_paths: set[str],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(package, mode="r:gz") as handle:
        members = handle.getmembers()
        require(len(members) == 28, "execution-package member cardinality")
        for member in members:
            name = safe_archive_name(member.name, "execution package")
            require(member.isfile(), "non-file execution-package member")
            require(member.name not in observed, "duplicate execution-package member")
            observed.add(member.name)
            require(member.name in expected_paths, "unexpected execution-package path")
            source = handle.extractfile(member)
            require(source is not None, "execution-package payload missing")
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), "execution-package extraction collision")
            with source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
    require(observed == expected_paths, "execution-package exact path set")


def verify_frozen_inputs(
    *,
    evidence_root: Path,
    package_extract_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    paths = {name: evidence_root / name for name in EXPECTED_HASHES if name != "r35-historical-alias-20260826a-output.tar.gz"}
    observed_hashes: dict[str, str] = {}
    for name, expected in EXPECTED_HASHES.items():
        if name == "r35-historical-alias-20260826a-output.tar.gz":
            continue
        observed = sha256_file(paths[name])
        require(observed == expected, f"frozen {name} hash drift")
        observed_hashes[name] = observed

    checksum_path = evidence_root / "r35-execution-package.tar.gz.sha256"
    require(checksum_path.is_file() and not checksum_path.is_symlink(), "package checksum receipt missing")
    checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    require(
        checksum_lines
        == [
            EXPECTED_HASHES["r35-execution-package.tar.gz"]
            + "  r35-execution-package.tar.gz"
        ],
        "package checksum receipt drift",
    )

    preregistration = load_json(paths["preregistration.json"], "preregistration")
    execution_input = load_json(paths["static-execution-input.json"], "execution input")
    amendment = load_json(paths["preexecution-resource-amendment.json"], "resource amendment")
    freeze = load_json(paths["preexecution-freeze-receipt.json"], "freeze receipt")
    require(preregistration.get("schema_version") == "forkaudit-r35-historical-alias-protocol-v1", "preregistration schema")
    require(preregistration.get("run_id") == RUN_ID and preregistration.get("rank_count") == 8, "preregistration run geometry")
    require(execution_input.get("schema_version") == "forkaudit-r35-historical-alias-execution-input-v1", "execution-input schema")
    require(execution_input.get("status") == "frozen_before_candidate_outputs", "execution-input freeze status")
    require(execution_input.get("candidate_output_seen_when_frozen") is False, "execution-input outcome blindness")
    require(execution_input.get("run_id") == RUN_ID and execution_input.get("rank_count") == 8, "execution-input run geometry")
    require(execution_input.get("protocol") == preregistration, "execution-input/preregistration binding")
    require(
        execution_input.get("output", {}).get("run_root")
        == "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r35-historical-alias-20260826a",
        "execution-input output root",
    )

    exact_keys(
        amendment,
        {
            "schema_version",
            "status",
            "created_at_utc",
            "run_id",
            "preregistration_raw_sha256",
            "execution_input_raw_sha256",
            "source_ledger_raw_sha256",
            "execution_package_sha256",
            "science_design_changed",
            "candidate_output_seen_when_frozen",
            "job_id",
            "trial_id",
            "pod",
            "gpu_assignments",
        },
        "resource amendment",
    )
    require(amendment["schema_version"] == "forkaudit-r35-resource-amendment-v1", "resource amendment schema")
    require(amendment["status"] == "frozen_after_resource_creation_before_candidate_outputs", "resource amendment status")
    require(amendment["run_id"] == RUN_ID, "resource amendment run id")
    require(amendment["science_design_changed"] is False, "resource amendment science drift")
    require(amendment["candidate_output_seen_when_frozen"] is False, "resource amendment outcome blindness")
    require(
        amendment["preregistration_raw_sha256"] == EXPECTED_HASHES["preregistration.json"]
        and amendment["execution_input_raw_sha256"] == EXPECTED_HASHES["static-execution-input.json"]
        and amendment["source_ledger_raw_sha256"] == EXPECTED_HASHES["source.sha256"]
        and amendment["execution_package_sha256"] == EXPECTED_HASHES["r35-execution-package.tar.gz"],
        "resource amendment hash bindings",
    )
    assignments = amendment["gpu_assignments"]
    require(isinstance(assignments, dict) and set(assignments) == {str(rank) for rank in range(8)}, "amendment GPU rank coverage")
    uuids: list[str] = []
    for rank in range(8):
        row = assignments[str(rank)]
        require(
            isinstance(row, dict)
            and set(row) == {"physical_index", "uuid"}
            and row == {"physical_index": rank, "uuid": row.get("uuid")}
            and isinstance(row.get("uuid"), str)
            and row["uuid"].startswith("GPU-"),
            f"amendment GPU assignment {rank}",
        )
        uuids.append(row["uuid"])
    require(len(set(uuids)) == 8, "amendment GPU UUID uniqueness")

    require(freeze.get("schema_version") == "forkaudit-r35-preexecution-freeze-receipt-v1", "freeze receipt schema")
    require(freeze.get("status") == "science_and_execution_package_frozen_before_candidate_outputs", "freeze receipt status")
    require(freeze.get("run_id") == RUN_ID and freeze.get("candidate_output_seen_when_frozen") is False, "freeze receipt binding")
    frozen_artifacts = freeze.get("artifacts")
    require(isinstance(frozen_artifacts, dict), "freeze artifact receipt")
    require(
        frozen_artifacts.get("preregistration_raw_sha256") == EXPECTED_HASHES["preregistration.json"]
        and frozen_artifacts.get("execution_input_raw_sha256") == EXPECTED_HASHES["static-execution-input.json"]
        and frozen_artifacts.get("source_ledger_raw_sha256") == EXPECTED_HASHES["source.sha256"]
        and frozen_artifacts.get("execution_package_sha256") == EXPECTED_HASHES["r35-execution-package.tar.gz"],
        "freeze artifact hash bindings",
    )

    ledger = parse_source_ledger(paths["source.sha256"])
    expected_package_paths = package_expected_paths(ledger)
    extract_frozen_package(
        paths["r35-execution-package.tar.gz"],
        package_extract_root,
        expected_package_paths,
    )
    for name, expected_sha in ledger.items():
        require(
            sha256_file(package_extract_root.joinpath(*PurePosixPath(name).parts))
            == expected_sha,
            f"source-ledger target drift: {PurePosixPath(name).name}",
        )
    evidence_prefix = PurePosixPath(
        "paper_autonomous_multifork_iteration/evidence/r35_historical_alias_regression"
    )
    for name in ("preregistration.json", "static-execution-input.json", "source.sha256"):
        packaged = package_extract_root.joinpath(*(evidence_prefix / name).parts)
        require(packaged.read_bytes() == paths[name].read_bytes(), f"packaged {name} drift")

    frozen = {
        name: {
            "path": f"evidence/r35_historical_alias_regression/{name}",
            "sha256": observed_hashes[name],
            "nbytes": paths[name].stat().st_size,
        }
        for name in sorted(observed_hashes)
    }
    frozen["r35-execution-package.tar.gz"]["member_count"] = 28
    frozen["r35-execution-package.tar.gz"]["source_ledger_row_count"] = 11
    frozen["r35-execution-package.tar.gz"]["all_source_ledger_entries_verified"] = True
    return preregistration, amendment, {"frozen": frozen, "uuids": uuids}


def read_fp32_sidecar(path: Path, expected_count: int) -> tuple[str, int]:
    payload = path.read_bytes()
    require(len(payload) == expected_count * 4, "sidecar byte geometry")
    values = array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    require(len(values) == expected_count, "sidecar element count")
    require(all(math.isfinite(float(value)) for value in values), "non-finite sidecar")
    require(bool(values), "empty sidecar")
    argmax = max(range(len(values)), key=values.__getitem__)
    return hashlib.sha256(payload).hexdigest(), argmax


def empty_aggregate_counts() -> dict[str, Any]:
    return {
        "cell_count": 0,
        "ranks": [],
        "hypothesis_true_counts": {name: 0 for name in OUTCOMES},
        "baseline_true_counts": {
            pair: {metric: 0 for metric in METRICS} for pair in PAIRS
        },
        "historical_gate_and_output_only_exact_count": 0,
        "historical_persistent_base_invariant_violation_count": 0,
    }


def aggregate_counts(replays: Sequence[Mapping[str, Any]], ranks: Sequence[int]) -> dict[str, Any]:
    by_rank = {int(row["rank"]): row for row in replays}
    output = empty_aggregate_counts()
    for rank in ranks:
        replay = by_rank[rank]
        outcomes = replay["hypothesis_outcomes"]
        matrix = replay["conventional_baseline_matrix"]
        output["cell_count"] += 1
        output["ranks"].append(rank)
        for name in OUTCOMES:
            output["hypothesis_true_counts"][name] += int(outcomes[name])
        for pair in PAIRS:
            for metric in METRICS:
                output["baseline_true_counts"][pair][metric] += int(matrix[pair][metric])
        output["historical_gate_and_output_only_exact_count"] += int(
            outcomes["historical_expected_authenticated_first_gate_reproduced"]
            and outcomes["historical_output_only_exact_to_materialized"]
        )
        output["historical_persistent_base_invariant_violation_count"] += int(
            not matrix["historical_pre_fix_vs_materialized_control"]
            ["persistent_base_content_only_invariant"]
        )
    return output


def coordinate_summary(replays: Sequence[Mapping[str, Any]], ranks: Sequence[int]) -> dict[str, Any]:
    by_rank = {int(row["rank"]): row for row in replays}
    historical_materialized = {metric: 0 for metric in METRICS}
    repaired_materialized = {metric: 0 for metric in METRICS}
    outcome_counts = {name: 0 for name in OUTCOMES}
    exact_gate_count = 0
    operational_count = 0
    for rank in ranks:
        replay = by_rank[rank]
        operational_count += int(replay["operational_valid"] is True)
        historical = replay["lanes"]["historical_pre_fix"]
        exact_gate_count += int(
            historical["completed_receipt_ids"] == list(RECEIPT_ORDER[:2])
            and historical["reported_first_gate"] == GATE
            and historical["row_recomputed_first_gate"] == GATE
        )
        for metric in METRICS:
            historical_materialized[metric] += int(
                replay["conventional_baseline_matrix"]
                ["historical_pre_fix_vs_materialized_control"][metric]
            )
            repaired_materialized[metric] += int(
                replay["conventional_baseline_matrix"]
                ["repaired_borrowed_vs_materialized_control"][metric]
            )
        for name in OUTCOMES:
            outcome_counts[name] += int(replay["hypothesis_outcomes"][name])
    count = len(ranks)
    positive = (
        exact_gate_count == count
        and historical_materialized["greedy_token_exact"] == count
        and historical_materialized["full_fp32_logits_exact"] == count
        and outcome_counts["repaired_storage_contract_clean"] == count
        and outcome_counts["materialized_storage_contract_clean"] == count
        and outcome_counts["repaired_semantic_and_terminal_exact_to_materialized"] == count
    )
    return {
        "cell_count": count,
        "ranks": list(ranks),
        "operational_valid_rank_count": operational_count,
        "all_cells_operationally_valid": operational_count == count,
        "historical_authenticated_receipt_and_inner_predicate_count": exact_gate_count,
        "historical_vs_materialized_true_counts": historical_materialized,
        "historical_base_immutability_violation_count": count
        - historical_materialized["persistent_base_content_only_invariant"],
        "repaired_audit_pass_count": outcome_counts["repaired_storage_contract_clean"],
        "materialized_audit_pass_count": outcome_counts["materialized_storage_contract_clean"],
        "repaired_vs_materialized_true_counts": repaired_materialized,
        "hypothesis_true_counts": outcome_counts,
        "registered_positive_headline_satisfied": positive,
    }


def verify_chain(
    *,
    run_root: Path,
    preregistration: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    launch = load_json(run_root / "launch-completion.json", "launch completion")
    aggregate = load_json(run_root / "aggregate.json", "aggregate")
    exact_keys(launch, {"schema_version", "status", "artifacts"}, "launch completion")
    require(launch["schema_version"] == "forkaudit-r35-launch-completion-v1", "launch schema")
    require(launch["status"] == "completed", "launch status")
    expected_launch_paths = {"aggregate.json"} | {
        f"replay/rank-{rank}-replay.json" for rank in range(8)
    }
    launch_rows = launch["artifacts"]
    require(isinstance(launch_rows, list) and len(launch_rows) == 9, "launch artifact cardinality")
    require(
        {row.get("path") for row in launch_rows if isinstance(row, dict)}
        == expected_launch_paths,
        "launch artifact path set",
    )
    for row in launch_rows:
        require(isinstance(row, dict) and set(row) == {"path", "sha256"}, "launch artifact row")
        check_sha(row["sha256"], "launch artifact")
        require(sha256_file(run_root / row["path"]) == row["sha256"], "launch artifact hash")

    require(aggregate.get("schema_version") == "forkaudit-r35-historical-alias-aggregate-v1", "aggregate schema")
    require(aggregate.get("run_id") == RUN_ID, "aggregate run id")
    require(aggregate.get("status") == "operationally_valid_aggregate" and aggregate.get("operational_valid") is True, "aggregate operational validity")
    require(
        aggregate.get("operational_cardinality")
        == {
            "rank_replay_count": 8,
            "lane_count": 24,
            "fp32_sidecar_count": 24,
            "unique_case_nonce_count": 24,
        },
        "aggregate operational cardinality",
    )
    require(
        aggregate.get("preregistration_raw_sha256") == EXPECTED_HASHES["preregistration.json"]
        and aggregate.get("execution_input_raw_sha256") == EXPECTED_HASHES["static-execution-input.json"]
        and aggregate.get("source_ledger_raw_sha256") == EXPECTED_HASHES["source.sha256"]
        and aggregate.get("amendment_raw_sha256") == EXPECTED_HASHES["preexecution-resource-amendment.json"],
        "aggregate frozen-input bindings",
    )
    require(
        aggregate.get("headline_outcomes") == {name: True for name in HEADLINES},
        "aggregate registered headline",
    )
    require(aggregate.get("population_detection_rate_computed") is False, "aggregate rate claim")
    require(aggregate.get("statistical_independence_claimed") is False, "aggregate independence claim")

    replay_hash_map = aggregate.get("rank_replay_raw_sha256")
    require(isinstance(replay_hash_map, dict) and set(replay_hash_map) == {str(rank) for rank in range(8)}, "aggregate replay hash map")
    replays: list[dict[str, Any]] = []
    rank_receipts: list[dict[str, Any]] = []
    nonces: set[str] = set()
    sidecar_keys: set[tuple[int, str]] = set()
    uuids: set[str] = set()
    for rank in range(8):
        replay_path = run_root / "replay" / f"rank-{rank}-replay.json"
        replay_sha = sha256_file(replay_path)
        require(replay_hash_map[str(rank)] == replay_sha, f"aggregate/replay hash rank {rank}")
        replay = load_json(replay_path, f"rank {rank} replay")
        replays.append(replay)
        require(replay.get("schema_version") == "forkaudit-r35-historical-alias-rank-replay-v1", f"rank {rank} replay schema")
        require(replay.get("run_id") == RUN_ID and replay.get("rank") == rank, f"rank {rank} replay identity")
        require(replay.get("status") == "operationally_valid_detached_replay" and replay.get("operational_valid") is True, f"rank {rank} replay validity")
        require(replay.get("candidate_modules_imported") is False, f"rank {rank} replay import isolation")
        require(replay.get("verified_lane_count") == 3 and replay.get("verified_sidecar_count") == 3, f"rank {rank} replay cardinality")
        require(replay.get("fixed_pair_mappings") == list(PAIRS), f"rank {rank} pair mapping")
        require(replay.get("population_detection_rate_computed") is False, f"rank {rank} rate claim")
        require(
            replay.get("preregistration_raw_sha256") == EXPECTED_HASHES["preregistration.json"]
            and replay.get("execution_input_raw_sha256") == EXPECTED_HASHES["static-execution-input.json"]
            and replay.get("source_ledger_raw_sha256") == EXPECTED_HASHES["source.sha256"]
            and replay.get("amendment_raw_sha256") == EXPECTED_HASHES["preexecution-resource-amendment.json"],
            f"rank {rank} replay frozen bindings",
        )
        result_path = run_root / f"rank-{rank}" / "raw" / "rank-result.json"
        result_sha = sha256_file(result_path)
        require(replay.get("rank_result_raw_sha256") == result_sha, f"rank {rank} replay/result hash")
        result = load_json(result_path, f"rank {rank} result")
        require(result.get("schema_version") == "forkaudit-r35-historical-alias-rank-v1", f"rank {rank} result schema")
        require(result.get("run_id") == RUN_ID and result.get("rank") == rank, f"rank {rank} result identity")
        require(result.get("status") == "rank_completed" and result.get("operational_invalid") is None, f"rank {rank} result status")
        require(result.get("scientific_outcome_does_not_control_operational_validity") is True, f"rank {rank} outcome/validity boundary")
        require(result.get("protocol") == preregistration, f"rank {rank} protocol binding")
        require(result.get("fault_isolation") == EXPECTED_FAULT_ISOLATION, f"rank {rank} fault isolation")
        assignment = amendment["gpu_assignments"][str(rank)]
        require(result.get("resource", {}).get("gpu_assignment") == assignment, f"rank {rank} resource assignment")
        require(result.get("resource", {}).get("execution_package_sha256") == EXPECTED_HASHES["r35-execution-package.tar.gz"], f"rank {rank} package binding")
        hardware = result.get("hardware")
        require(isinstance(hardware, dict), f"rank {rank} hardware receipt")
        require(
            hardware.get("physical_index") == rank
            and hardware.get("uuid") == assignment["uuid"]
            and hardware.get("cuda_visible_devices") == assignment["uuid"]
            and hardware.get("name") == "NVIDIA H20-3e",
            f"rank {rank} H20/UUID binding",
        )
        require(assignment["uuid"] not in uuids, "duplicate observed GPU UUID")
        uuids.add(assignment["uuid"])
        require(
            result.get("input_receipt", {}).get("coordinate_class")
            == ("archived" if rank < 3 else "additional_frozen"),
            f"rank {rank} coordinate class",
        )
        require(
            result.get("lane_order")
            == (list(LANES) if rank % 2 == 0 else list(reversed(LANES))),
            f"rank {rank} lane order",
        )
        result_lanes = result.get("lanes")
        replay_lanes = replay.get("lanes")
        require(isinstance(result_lanes, dict) and set(result_lanes) == set(LANES), f"rank {rank} result lanes")
        require(isinstance(replay_lanes, dict) and set(replay_lanes) == set(LANES), f"rank {rank} replay lanes")
        sidecar_rows: list[dict[str, Any]] = []
        for lane in LANES:
            result_lane = result_lanes[lane]
            replay_lane = replay_lanes[lane]
            require(result_lane.get("mutation_receipt") == EXPECTED_LANE_MUTATION, f"rank {rank} {lane} no mutation")
            nonce = replay_lane.get("case_nonce")
            require(nonce == result_lane.get("case_nonce") and isinstance(nonce, str), f"rank {rank} {lane} nonce binding")
            require(re.fullmatch(r"[0-9a-f]{32}|[0-9a-f]{64}", nonce) is not None, f"rank {rank} {lane} nonce format")
            require(nonce not in nonces, "duplicate global case nonce")
            nonces.add(nonce)
            sidecar = replay_lane.get("sidecar")
            producer_sidecar = result_lane.get("full_logits")
            require(isinstance(sidecar, dict) and isinstance(producer_sidecar, dict), f"rank {rank} {lane} sidecar receipt")
            expected_name = f"{lane}-full-fp32-logits.bin"
            require(sidecar.get("path") == expected_name, f"rank {rank} {lane} sidecar path")
            require(
                {key: sidecar[key] for key in ("path", "sha256", "dtype", "shape", "nbytes", "finite")}
                == producer_sidecar,
                f"rank {rank} {lane} producer/replay sidecar binding",
            )
            require(
                sidecar.get("dtype") == "float32-little-endian"
                and sidecar.get("shape") == [1, 248320]
                and sidecar.get("nbytes") == 993280
                and sidecar.get("finite") is True,
                f"rank {rank} {lane} sidecar contract",
            )
            key = (rank, expected_name)
            require(key not in sidecar_keys, "duplicate rank-qualified sidecar")
            sidecar_keys.add(key)
            sidecar_path = result_path.parent / expected_name
            observed_sha, argmax = read_fp32_sidecar(sidecar_path, 248320)
            require(observed_sha == sidecar.get("sha256"), f"rank {rank} {lane} sidecar hash")
            require(argmax == sidecar.get("argmax_token_id"), f"rank {rank} {lane} sidecar argmax")
            require(
                result_lane.get("model_step", {}).get("full_logit_sha256") == observed_sha
                and result_lane.get("model_step", {}).get("greedy_token_id") == argmax,
                f"rank {rank} {lane} model/sidecar binding",
            )
            sidecar_rows.append(
                {
                    "lane": lane,
                    "path": f"rank-{rank}/raw/{expected_name}",
                    "sha256": observed_sha,
                    "nbytes": 993280,
                    "finite": True,
                    "greedy_token_id": argmax,
                }
            )
        historical = replay_lanes["historical_pre_fix"]
        require(
            historical.get("completed_receipt_ids") == list(RECEIPT_ORDER[:2])
            and historical.get("reported_first_gate") == GATE
            and historical.get("row_recomputed_first_gate") == GATE,
            f"rank {rank} historical authenticated gate",
        )
        require(
            replay["hypothesis_outcomes"]
            == {
                "historical_expected_authenticated_first_gate_reproduced": True,
                "historical_output_only_exact_to_materialized": True,
                "repaired_storage_contract_clean": True,
                "materialized_storage_contract_clean": True,
                "repaired_semantic_and_terminal_exact_to_materialized": True,
            },
            f"rank {rank} registered outcomes",
        )
        rank_receipts.append(
            {
                "rank": rank,
                "coordinate_class": "archived" if rank < 3 else "additional_frozen",
                "gpu_uuid": assignment["uuid"],
                "rank_result_sha256": result_sha,
                "replay_sha256": replay_sha,
                "fault_isolation_verified": True,
                "sidecars": sidecar_rows,
            }
        )
    require(len(nonces) == 24 and len(sidecar_keys) == 24, "global nonce/sidecar cardinality")
    require(len(uuids) == 8, "global GPU UUID cardinality")

    archived_counts = aggregate_counts(replays, [0, 1, 2])
    additional_counts = aggregate_counts(replays, [3, 4, 5, 6, 7])
    all_counts = aggregate_counts(replays, list(range(8)))
    require(
        aggregate.get("coordinate_classes")
        == {
            "archived_coordinates": archived_counts,
            "additional_frozen_inputs": additional_counts,
        },
        "aggregate coordinate-class counts",
    )
    require(aggregate.get("overall_cell_counts") == all_counts, "aggregate all-cell counts")
    coordinate_counts = {
        "archived_coordinates": coordinate_summary(replays, [0, 1, 2]),
        "additional_frozen_inputs": coordinate_summary(replays, [3, 4, 5, 6, 7]),
        "all_coordinates": coordinate_summary(replays, list(range(8))),
    }
    return aggregate, replays, rank_receipts, coordinate_counts


def run_checked(command: Sequence[str], *, cwd: Path, label: str) -> None:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "LC_ALL": "C",
        "LANG": "C",
    }
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = "" if not detail else ": " + detail[-1]
        raise VerificationError(f"{label} failed{suffix}")


def rerun_detached_verifiers(
    *,
    package_root: Path,
    run_root: Path,
    amendment_path: Path,
    temporary_root: Path,
) -> dict[str, Any]:
    paper = package_root / "paper_autonomous_multifork_iteration"
    replay_script = paper / "scripts" / "r35_replay_historical_alias_regression.py"
    aggregate_script = paper / "scripts" / "r35_aggregate_historical_alias_regression.py"
    preregistration = paper / "evidence" / "r35_historical_alias_regression" / "preregistration.json"
    replay_root = temporary_root / "replay"
    replay_root.mkdir(parents=True, exist_ok=False)
    replay_hashes: dict[str, str] = {}
    for rank in range(8):
        output = replay_root / f"rank-{rank}-replay.json"
        run_checked(
            [
                sys.executable,
                "-I",
                str(replay_script),
                "--rank-result",
                str(run_root / f"rank-{rank}" / "raw" / "rank-result.json"),
                "--raw-root",
                str(run_root / f"rank-{rank}" / "raw"),
                "--preregistration",
                str(preregistration),
                "--expected-preregistration-sha256",
                EXPECTED_HASHES["preregistration.json"],
                "--amendment",
                str(amendment_path),
                "--expected-amendment-sha256",
                EXPECTED_HASHES["preexecution-resource-amendment.json"],
                "--output",
                str(output),
            ],
            cwd=package_root,
            label=f"detached replay rank {rank}",
        )
        archived = run_root / "replay" / f"rank-{rank}-replay.json"
        require(output.read_bytes() == archived.read_bytes(), f"rank {rank} replay byte mismatch")
        replay_hashes[str(rank)] = sha256_file(output)
    recomputed_aggregate = temporary_root / "aggregate.json"
    run_checked(
        [
            sys.executable,
            "-I",
            str(aggregate_script),
            "--replay-root",
            str(replay_root),
            "--expected-preregistration-sha256",
            EXPECTED_HASHES["preregistration.json"],
            "--expected-amendment-sha256",
            EXPECTED_HASHES["preexecution-resource-amendment.json"],
            "--expected-execution-input-sha256",
            EXPECTED_HASHES["static-execution-input.json"],
            "--expected-source-ledger-sha256",
            EXPECTED_HASHES["source.sha256"],
            "--output",
            str(recomputed_aggregate),
        ],
        cwd=package_root,
        label="detached aggregate",
    )
    archived_aggregate = run_root / "aggregate.json"
    require(recomputed_aggregate.read_bytes() == archived_aggregate.read_bytes(), "aggregate byte mismatch")
    return {
        "frozen_replay_program_sha256": sha256_file(replay_script),
        "frozen_aggregate_program_sha256": sha256_file(aggregate_script),
        "rank_replay_count": 8,
        "rank_replay_bytes_exact": True,
        "rank_replay_sha256": replay_hashes,
        "aggregate_bytes_exact": True,
        "aggregate_sha256": sha256_file(recomputed_aggregate),
        "candidate_or_runtime_modules_imported_by_verifier": False,
    }


def verify(
    *,
    paper_root: Path,
    evidence_root: Path,
    formal_root: Path,
) -> dict[str, Any]:
    run_root = formal_root / ARCHIVE_ROOT_NAME
    output_archive = formal_root / f"{ARCHIVE_ROOT_NAME}-output.tar.gz"
    archive_receipt = verify_output_archive(output_archive, run_root)
    with tempfile.TemporaryDirectory(prefix="r35-evidence-verification-") as temporary:
        temporary_root = Path(temporary)
        package_root = temporary_root / "package"
        preregistration, amendment, frozen_receipt = verify_frozen_inputs(
            evidence_root=evidence_root,
            package_extract_root=package_root,
        )
        aggregate, replays, rank_receipts, coordinate_counts = verify_chain(
            run_root=run_root,
            preregistration=preregistration,
            amendment=amendment,
        )
        detached = rerun_detached_verifiers(
            package_root=package_root,
            run_root=run_root,
            amendment_path=evidence_root / "preexecution-resource-amendment.json",
            temporary_root=temporary_root / "recomputed",
        )
    require(len(replays) == 8, "verified replay cardinality")
    return {
        "schema_version": SCHEMA,
        "status": "verified_fail_closed",
        "operational_valid": True,
        "run_id": RUN_ID,
        "validator": {
            "path": "scripts/validate_r35_historical_alias_evidence.py",
            "sha256": sha256_file(Path(__file__).resolve()),
            "standard_library_only": True,
            "postexecution_verifier_not_in_frozen_execution_package": True,
        },
        "artifact_roots": {
            "frozen_evidence": relative_to(evidence_root, paper_root),
            "formal_h20": relative_to(formal_root, paper_root),
            "extracted_run": relative_to(run_root, paper_root),
        },
        "frozen_inputs": frozen_receipt["frozen"],
        "output_archive": {
            **archive_receipt,
            "path": relative_to(output_archive, paper_root),
        },
        "evidence_graph": {
            "launch_completion_sha256": sha256_file(run_root / "launch-completion.json"),
            "aggregate_sha256": sha256_file(run_root / "aggregate.json"),
            "launch_artifact_count": 9,
            "rank_result_count": 8,
            "rank_replay_count": 8,
            "lane_count": 24,
            "fp32_sidecar_count": 24,
            "unique_case_nonce_count": 24,
            "unique_gpu_uuid_count": len(frozen_receipt["uuids"]),
            "all_hash_edges_verified": True,
            "all_sidecars_size_shape_finite_verified": True,
            "all_rank_status_uuid_fault_isolation_verified": True,
        },
        "ranks": rank_receipts,
        "coordinate_cell_counts": coordinate_counts,
        "headline_outcomes": aggregate["headline_outcomes"],
        "detached_reexecution": detached,
        "claim_boundary": {
            "eight_rows_are_coordinate_cells_not_eight_natural_bugs": True,
            "archived_and_additional_coordinates_reported_separately": True,
            "normalized_storage_ids_compared_within_lane_only": True,
            "normalized_storage_ids_compared_across_lanes": False,
            "population_detection_rate_computed": False,
            "statistical_independence_claimed": False,
        },
    }


def write_deterministic(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_bytes(value) + b"\n"
    if path.exists():
        require(path.is_file() and not path.is_symlink(), "verification output is not a regular file")
        require(path.read_bytes() == payload, "existing verification result differs")
        return hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    require(not pending.exists(), "stale verification output pending file")
    with pending.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)
    return hashlib.sha256(payload).hexdigest()


def parser() -> argparse.ArgumentParser:
    script = Path(__file__).resolve()
    paper_root = script.parents[1]
    evidence_root = paper_root / "evidence" / "r35_historical_alias_regression"
    formal_root = evidence_root / "formal_h20"
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--paper-root", type=Path, default=paper_root)
    value.add_argument("--evidence-root", type=Path, default=evidence_root)
    value.add_argument("--formal-root", type=Path, default=formal_root)
    value.add_argument(
        "--output",
        type=Path,
        default=formal_root / "RESULT_VERIFICATION.json",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        receipt = verify(
            paper_root=args.paper_root.resolve(),
            evidence_root=args.evidence_root.resolve(),
            formal_root=args.formal_root.resolve(),
        )
        receipt_sha = write_deterministic(args.output.resolve(), receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "output": relative_to(args.output.resolve(), args.paper_root.resolve()),
                    "sha256": receipt_sha,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (OSError, subprocess.SubprocessError, tarfile.TarError, VerificationError) as exc:
        print(f"R35 evidence verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
