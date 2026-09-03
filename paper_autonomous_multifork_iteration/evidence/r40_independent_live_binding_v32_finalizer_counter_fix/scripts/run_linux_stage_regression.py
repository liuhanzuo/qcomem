from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

import stage_v6_clean as stage


MODULES = (
    "test_qcomem_forkaudit_storage_witness",
    "test_qcomem_forkaudit_oracle",
    "test_qcomem_forkaudit_mutants",
    "test_qcomem_vllm_paged_multifork_resident",
    "test_qcomem_qwen35_vllm_paged_integration",
    "test_build_qcomem_forkaudit_rr2_input_manifest",
    "test_build_qcomem_forkaudit_fp32_calibration_manifest",
    "test_run_qcomem_qwen35_forkaudit_review_revision",
    "test_launch_qcomem_qwen35_forkaudit_review_revision",
)
PRIMARY_RELATIVE = Path(
    "paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/gpu"
)
EXACT_ERROR = "archive must contain exactly the eight expected raw shard JSON files"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_suite(python: Path, tree: Path, log: Path) -> dict[str, object]:
    code = tree / PRIMARY_RELATIVE
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=str(code),
    )
    command = [str(python), "-B", "-m", "unittest", "-v", *MODULES]
    with log.open("xb") as stream:
        result = subprocess.run(command, cwd=code, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text(encoding="utf-8", errors="strict")
    match = re.search(r"^Ran (\d+) tests in ", text, flags=re.MULTILINE)
    skips = re.search(r"skipped=(\d+)", text)
    return {
        "exit_code": result.returncode,
        "tests_ran": int(match.group(1)) if match else None,
        "error_headers": len(re.findall(r"^ERROR: ", text, flags=re.MULTILINE)),
        "failure_headers": len(re.findall(r"^FAIL: ", text, flags=re.MULTILINE)),
        "skipped": int(skips.group(1)) if skips else 0,
        "ok_terminal": bool(re.search(r"^OK(?: \(.*\))?$", text, flags=re.MULTILINE)),
        "exact_calibration_error_present": EXACT_ERROR in text,
        "log_sha256": stage.file_sha256(log),
    }


def run_formal_preflight(
    python: Path,
    tree: Path,
    v6_archive: Path,
    overlay_archive: Path,
    log: Path,
) -> dict[str, object]:
    package = tree / "paper_autonomous_multifork_iteration/evidence/r40_independent_live_binding_v32_finalizer_counter_fix"
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        R40_V32_REPO_ROOT=str(tree),
        R40_V32_CANONICAL_V6_ARCHIVE=str(v6_archive),
        R40_V32_OVERLAY_ARCHIVE=str(overlay_archive),
    )
    command = [
        str(python),
        "-B",
        str(package / "executed_source/r40_formal_preflight.py"),
        "--root",
        str(package),
        "--repo-root",
        str(tree),
    ]
    with log.open("xb") as stream:
        result = subprocess.run(command, cwd=package, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text(encoding="utf-8", errors="strict")
    match = re.search(r"^Ran (\d+) tests in ", text, flags=re.MULTILINE)
    skips = re.search(r"skipped=(\d+)", text)
    return {
        "exit_code": result.returncode,
        "tests_ran": int(match.group(1)) if match else None,
        "error_headers": len(re.findall(r"^ERROR: ", text, flags=re.MULTILINE)),
        "failure_headers": len(re.findall(r"^FAIL: ", text, flags=re.MULTILINE)),
        "skipped": int(skips.group(1)) if skips else 0,
        "ok_terminal": bool(re.search(r"^OK$", text, flags=re.MULTILINE)),
        "self_contained_package_present": package.is_dir(),
        "v28_sibling_absent": not (package.parent / "r40_independent_live_binding_v28_published_phase_path_fix").exists(),
        "log_sha256": stage.file_sha256(log),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--v6-archive", type=Path, required=True)
    parser.add_argument("--overlay-archive", type=Path, required=True)
    parser.add_argument("--overlay-sha256", required=True)
    parser.add_argument("--clean-ledger", type=Path, required=True)
    parser.add_argument("--exclusion-ledger", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work, _ = stage.normalize_new_output(args.work_root)
    work.mkdir(mode=0o700)
    raw = work / "raw-python-tarfile"
    raw.mkdir()
    stage.verified_ledgers(args.v6_archive, args.clean_ledger, args.exclusion_ledger)
    with tarfile.open(args.v6_archive, "r:gz") as archive:
        archive.extractall(raw)
    raw_appledouble = [path for path in raw.rglob("._*") if path.is_file()]
    require(len(raw_appledouble) == 130, "raw extraction did not reproduce exact 130-member AppleDouble materialization")
    raw_result = run_suite(args.python, raw, work / "raw-162.log")
    require(raw_result["tests_ran"] == 162, "raw regression did not run 162 tests")
    require(raw_result["error_headers"] == 2 and raw_result["failure_headers"] == 0, "raw regression did not reproduce exactly two errors")
    require(raw_result["exact_calibration_error_present"] is True, "raw regression exact v16 failure absent")
    clean = work / "qcomem_r40_v32_finalizer_counter_fix_20260902a"
    receipt = stage.prepare_stage(
        output_root=clean,
        v6_archive=args.v6_archive,
        overlay_archive=args.overlay_archive,
        clean_ledger=args.clean_ledger,
        exclusion_ledger=args.exclusion_ledger,
        expected_v6_sha256=stage.V6_ARCHIVE_SHA256,
        expected_overlay_sha256=args.overlay_sha256,
    )
    require(not list(clean.rglob("._*")), "clean stage contains AppleDouble path")
    clean_result = run_suite(args.python, clean, work / "clean-162.log")
    require(clean_result["tests_ran"] == 162, "clean regression did not run 162 tests")
    require(clean_result["exit_code"] == 0 and clean_result["ok_terminal"] is True, "clean 162-test regression failed")
    require(clean_result["error_headers"] == 0 and clean_result["failure_headers"] == 0, "clean regression contains error/failure")
    require(clean_result["skipped"] == 0, "clean regression requires zero skip")
    formal_preflight = run_formal_preflight(
        args.python,
        clean,
        args.v6_archive,
        args.overlay_archive,
        work / "clean-formal-preflight-32.log",
    )
    require(formal_preflight["tests_ran"] == 32, "clean formal preflight did not run exactly 32 package tests")
    require(formal_preflight["exit_code"] == 0 and formal_preflight["ok_terminal"] is True, "clean formal preflight failed")
    require(formal_preflight["error_headers"] == 0 and formal_preflight["failure_headers"] == 0, "clean formal preflight contains error/failure")
    require(formal_preflight["skipped"] == 0, "clean formal preflight requires zero skip")
    require(formal_preflight["v28_sibling_absent"] is True, "clean formal preflight unexpectedly contains v28 sibling package")
    value = {
        "schema_version": "forkaudit-r40-v32-finalizer-counter-fix-stage-executable-regression-v1",
        "status": stage.STATUS,
        "canonical_v6_archive_sha256": stage.V6_ARCHIVE_SHA256,
        "overlay_archive_sha256": args.overlay_sha256,
        "raw_appledouble_paths": len(raw_appledouble),
        "raw": raw_result,
        "clean": clean_result,
        "clean_formal_preflight": formal_preflight,
        "stage_receipt_payload_sha256": receipt["payload_sha256"],
        "formal_gpu_execution": "not-run",
    }
    output = Path(os.path.abspath(os.fspath(args.output)))
    require(output.parent == work and not os.path.lexists(output), "regression output must be new and directly inside work root")
    stage.write_json_exclusive(output, value)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
