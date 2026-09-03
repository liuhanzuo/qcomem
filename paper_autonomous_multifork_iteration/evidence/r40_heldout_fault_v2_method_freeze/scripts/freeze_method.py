"""Freeze validated method-v2 bytes into a deterministic nonoverwriting package."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import sys
import tarfile


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT / "executed_source"))
from v2_common import (  # noqa: E402
    require,
    sha256_file,
    write_new_bytes,
    write_new_json,
)


PACKAGE_RELATIVE = Path("packages/r40-method-v2-freeze-20260827a.tar.gz")


def excluded(path: Path) -> bool:
    relative = path.relative_to(METHOD_ROOT)
    return (
        "__pycache__" in relative.parts
        or path.suffix == ".pyc"
        or relative.parts[0] == "packages"
        or relative.as_posix() in {
            "source-code.sha256", "method-freeze.json", "package-manifest.json",
            "TERMINAL_SHA256SUMS", "METHOD_FROZEN.json",
        }
        or path.name.startswith(".")
    )


def source_files() -> list[Path]:
    return sorted(
        (path for path in METHOD_ROOT.rglob("*") if path.is_file() and not excluded(path)),
        key=lambda path: path.relative_to(METHOD_ROOT).as_posix(),
    )


def manifest_bytes(paths: list[Path]) -> bytes:
    return "".join(
        "%s  %s\n" % (sha256_file(path), path.relative_to(METHOD_ROOT).as_posix())
        for path in paths
    ).encode("utf-8")


def write_deterministic_archive(path: Path, paths: list[Path]) -> None:
    require(not path.exists(), "package already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name("." + path.name + ".pending")
    require(not pending.exists(), "stale package pending file")
    with pending.open("xb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                for source in paths:
                    relative = source.relative_to(METHOD_ROOT)
                    info = archive.gettarinfo(str(source), arcname=(METHOD_ROOT.name + "/" + relative.as_posix()))
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o755 if source.suffix == ".sh" else 0o644
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    pending.replace(path)


def verify_archive(path: Path, paths: list[Path]) -> None:
    expected = {
        METHOD_ROOT.name + "/" + source.relative_to(METHOD_ROOT).as_posix(): sha256_file(source)
        for source in paths
    }
    observed: dict[str, str] = {}
    import hashlib
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            require(member.isfile() and member.name in expected, "unexpected archive member")
            handle = archive.extractfile(member)
            require(handle is not None, "archive member unreadable")
            observed[member.name] = hashlib.sha256(handle.read()).hexdigest()
    require(observed == expected, "archive member/hash mismatch")


def main() -> int:
    require(not (METHOD_ROOT / "METHOD_FROZEN.json").exists(), "method already frozen")
    validation_path = METHOD_ROOT / "local-validation.json"
    require(validation_path.is_file(), "local validation missing")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    require(validation.get("status") == "PASS_METHOD_FREEZE_ONLY", "local validation did not pass")
    require(validation.get("v2_fault_set_exists") is False, "fault set contamination")
    require(validation.get("h20_execution_performed") is False, "unexpected H20 result")

    blocker = {
        "schema_version": "forkaudit-method-v2-formal-blocker-v1",
        "local_method_status": "PASS_METHOD_FREEZE_ONLY",
        "scientific_campaign_status": "HOLD",
        "blocking_conditions": [
            "independent audit of frozen method and trust boundary not complete",
            "H20 live-state reader and injection runner not clean-validated or frozen",
            "fresh isolated designer has not received the frozen public snapshot",
            "eight-case fault set does not exist",
            "formal configuration is null-bound and unauthorized",
            "no one-shot H20 execution or frozen-verifier result exists"
        ],
        "paper_claim_authorized": False,
    }
    write_new_json(METHOD_ROOT / "formal-blocker.json", blocker)

    paths = source_files()
    ledger = manifest_bytes(paths)
    write_new_bytes(METHOD_ROOT / "source-code.sha256", ledger)
    ledger_sha = sha256_file(METHOD_ROOT / "source-code.sha256")
    snapshot_sha = sha256_file(METHOD_ROOT / "designer_snapshot/SHA256SUMS")
    prereg = json.loads((METHOD_ROOT / "preregistration.json").read_text(encoding="utf-8"))
    require(prereg["future_fault_freeze"]["designer_snapshot_sha256"] == snapshot_sha,
            "snapshot binding drift")
    freeze = {
        "schema_version": "forkaudit-method-v2-method-freeze-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_METHOD_FREEZE_ONLY",
        "source_ledger_sha256": ledger_sha,
        "source_file_count": len(paths),
        "contract_sha256": sha256_file(METHOD_ROOT / "METHOD_V2_CONTRACT.md"),
        "preregistration_sha256": sha256_file(METHOD_ROOT / "preregistration.json"),
        "predicate_source_sha256": sha256_file(METHOD_ROOT / "executed_source/v2_predicates.py"),
        "capture_source_sha256": sha256_file(METHOD_ROOT / "executed_source/v2_capture.py"),
        "integration_source_sha256": sha256_file(METHOD_ROOT / "executed_source/v2_integration.py"),
        "designer_snapshot_manifest_sha256": snapshot_sha,
        "local_validation_sha256": sha256_file(validation_path),
        "development_inputs_manifest_sha256": sha256_file(METHOD_ROOT / "development-inputs.sha256"),
        "development_inputs_are_scoring": False,
        "fault_set_sha256": None,
        "h20_execution_result_sha256": None,
        "detector_changes_after_this_freeze_allowed": False,
        "future_classification_if_protocol_completed": "method-v2-held-out-designer-executor-separated-constructed-cases",
        "positive_scientific_claim_authorized": False,
        "campaign_status": "HOLD_PENDING_INDEPENDENT_AUDIT_AND_FRESH_FAULT_FREEZE",
        "acceptance_gates": {key: "PASS" for key in ("L01", "L02", "L03", "L04", "L05", "L06", "L07")},
    }
    write_new_json(METHOD_ROOT / "method-freeze.json", freeze)

    archive_members = source_files() + [METHOD_ROOT / "source-code.sha256", METHOD_ROOT / "method-freeze.json"]
    archive_members = sorted(set(archive_members), key=lambda path: path.relative_to(METHOD_ROOT).as_posix())
    package = METHOD_ROOT / PACKAGE_RELATIVE
    write_deterministic_archive(package, archive_members)
    verify_archive(package, archive_members)
    package_manifest = {
        "schema_version": "forkaudit-method-v2-package-manifest-v1",
        "package_path": PACKAGE_RELATIVE.as_posix(),
        "package_sha256": sha256_file(package),
        "member_count": len(archive_members),
        "source_ledger_sha256": ledger_sha,
        "method_freeze_sha256": sha256_file(METHOD_ROOT / "method-freeze.json"),
        "deterministic_tar": {"uid": 0, "gid": 0, "mtime": 0, "gzip_mtime": 0},
    }
    write_new_json(METHOD_ROOT / "package-manifest.json", package_manifest)
    marker = {
        "schema_version": "forkaudit-method-v2-frozen-marker-v1",
        "status": "FROZEN_DO_NOT_EDIT",
        "source_ledger_sha256": ledger_sha,
        "method_freeze_sha256": sha256_file(METHOD_ROOT / "method-freeze.json"),
        "package_sha256": sha256_file(package),
        "scientific_campaign_status": "HOLD",
    }
    write_new_json(METHOD_ROOT / "METHOD_FROZEN.json", marker)
    terminals = [
        Path("source-code.sha256"), Path("method-freeze.json"), Path("package-manifest.json"),
        Path("METHOD_FROZEN.json"), Path("local-validation.json"),
        Path("development-regression.json"), Path("designer_snapshot/SHA256SUMS"),
        PACKAGE_RELATIVE,
    ]
    terminal_bytes = "".join(
        "%s  %s\n" % (sha256_file(METHOD_ROOT / relative), relative.as_posix())
        for relative in terminals
    ).encode("utf-8")
    write_new_bytes(METHOD_ROOT / "TERMINAL_SHA256SUMS", terminal_bytes)
    print(json.dumps({
        "status": freeze["status"],
        "campaign_status": freeze["campaign_status"],
        "source_ledger_sha256": ledger_sha,
        "method_freeze_sha256": sha256_file(METHOD_ROOT / "method-freeze.json"),
        "designer_snapshot_manifest_sha256": snapshot_sha,
        "package_sha256": sha256_file(package),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
