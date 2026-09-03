"""Freeze validated method-v3 bytes into a deterministic package."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT / "executed_source"))
from v3_common import require, sha256_file, write_new_bytes, write_new_json  # noqa: E402


PACKAGE_RELATIVE = Path("packages/r40-method-v3-freeze-20260827a.tar.gz")


def excluded(path: Path) -> bool:
    relative = path.relative_to(METHOD_ROOT)
    return (
        "__pycache__" in relative.parts or path.suffix == ".pyc"
        or relative.parts[0] == "packages" or path.name.startswith(".")
        or relative.as_posix() in {
            "source-code.sha256", "method-freeze.json", "package-manifest.json",
            "METHOD_FROZEN.json", "TERMINAL_SHA256SUMS",
        }
    )


def source_files() -> list[Path]:
    return sorted(
        (path for path in METHOD_ROOT.rglob("*") if path.is_file() and not excluded(path)),
        key=lambda path: path.relative_to(METHOD_ROOT).as_posix(),
    )


def ledger_bytes(paths: list[Path]) -> bytes:
    return "".join(
        "%s  %s\n" % (sha256_file(path), path.relative_to(METHOD_ROOT).as_posix())
        for path in paths
    ).encode("utf-8")


def write_archive(path: Path, members: list[Path]) -> None:
    require(not path.exists(), "package exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name("." + path.name + ".pending")
    with pending.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for source in members:
                    relative = source.relative_to(METHOD_ROOT)
                    info = archive.gettarinfo(str(source), METHOD_ROOT.name + "/" + relative.as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o755 if source.suffix == ".sh" else 0o644
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)
        raw.flush()
        os.fsync(raw.fileno())
    pending.replace(path)


def verify_archive(path: Path, members: list[Path]) -> None:
    expected = {
        METHOD_ROOT.name + "/" + source.relative_to(METHOD_ROOT).as_posix(): sha256_file(source)
        for source in members
    }
    observed = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            require(member.isfile() and member.name in expected, "archive member")
            handle = archive.extractfile(member)
            require(handle is not None, "archive read")
            observed[member.name] = hashlib.sha256(handle.read()).hexdigest()
    require(observed == expected, "archive hashes")


def main() -> int:
    require(not (METHOD_ROOT / "METHOD_FROZEN.json").exists(), "already frozen")
    validation = json.loads((METHOD_ROOT / "local-validation.json").read_text())
    require(validation["status"] == "PASS_METHOD_FREEZE_ONLY", "validation status")
    require(validation["v3_fault_set_exists"] is False and validation["h20_execution_performed"] is False,
            "unexpected scientific input")
    blocker = {
        "schema_version": "forkaudit-method-v3-formal-blocker-v1",
        "local_method_status": "PASS_METHOD_FREEZE_ONLY",
        "scientific_campaign_status": "HOLD",
        "blocking_conditions": [
            "fresh independent audit of the exact v3 freeze is not complete",
            "fresh isolated designer has not received the v3 public snapshot",
            "v3 fault set and audited runner bundle do not exist",
            "fixed formal configuration is absent and unauthorized",
            "no eight-H20 execution or frozen-verifier result exists"
        ],
        "paper_claim_authorized": False,
    }
    write_new_json(METHOD_ROOT / "formal-blocker.json", blocker)
    paths = source_files()
    write_new_bytes(METHOD_ROOT / "source-code.sha256", ledger_bytes(paths))
    freeze = {
        "schema_version": "forkaudit-method-v3-method-freeze-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_METHOD_FREEZE_ONLY",
        "source_ledger_sha256": sha256_file(METHOD_ROOT / "source-code.sha256"),
        "source_file_count": len(paths),
        "method_core_manifest_sha256": sha256_file(METHOD_ROOT / "method-core.sha256"),
        "authoritative_config_sha256": sha256_file(METHOD_ROOT / "authoritative_config.json"),
        "preregistration_sha256": sha256_file(METHOD_ROOT / "preregistration.json"),
        "schedule_sha256": sha256_file(METHOD_ROOT / "schedule.json"),
        "atomic_policy_sha256": sha256_file(METHOD_ROOT / "atomic_policy.json"),
        "designer_snapshot_manifest_sha256": sha256_file(METHOD_ROOT / "designer_snapshot/SHA256SUMS"),
        "local_validation_sha256": sha256_file(METHOD_ROOT / "local-validation.json"),
        "fault_set_sha256": None, "formal_config_sha256": None,
        "h20_result_sha256": None, "detector_changes_after_freeze_allowed": False,
        "acceptance_gates": {"L%02d" % index: "PASS" for index in range(1, 11)},
        "positive_scientific_claim_authorized": False,
        "campaign_status": "HOLD_PENDING_FRESH_INDEPENDENT_AUDIT",
    }
    write_new_json(METHOD_ROOT / "method-freeze.json", freeze)
    members = source_files() + [METHOD_ROOT / "source-code.sha256", METHOD_ROOT / "method-freeze.json"]
    members = sorted(set(members), key=lambda path: path.relative_to(METHOD_ROOT).as_posix())
    package = METHOD_ROOT / PACKAGE_RELATIVE
    write_archive(package, members)
    verify_archive(package, members)
    package_manifest = {
        "schema_version": "forkaudit-method-v3-package-manifest-v1",
        "package_path": PACKAGE_RELATIVE.as_posix(), "package_sha256": sha256_file(package),
        "member_count": len(members),
        "source_ledger_sha256": freeze["source_ledger_sha256"],
        "method_freeze_sha256": sha256_file(METHOD_ROOT / "method-freeze.json"),
        "deterministic_tar": {"uid": 0, "gid": 0, "mtime": 0, "gzip_mtime": 0},
    }
    write_new_json(METHOD_ROOT / "package-manifest.json", package_manifest)
    marker = {
        "schema_version": "forkaudit-method-v3-frozen-marker-v1",
        "status": "FROZEN_DO_NOT_EDIT",
        "source_ledger_sha256": freeze["source_ledger_sha256"],
        "method_freeze_sha256": sha256_file(METHOD_ROOT / "method-freeze.json"),
        "package_sha256": sha256_file(package), "scientific_campaign_status": "HOLD",
    }
    write_new_json(METHOD_ROOT / "METHOD_FROZEN.json", marker)
    terminal_members = (
        Path("source-code.sha256"), Path("method-freeze.json"), Path("package-manifest.json"),
        Path("METHOD_FROZEN.json"), Path("local-validation.json"),
        Path("audit-counterexample-results.json"), Path("method-core.sha256"),
        Path("authoritative_config.json"), Path("designer_snapshot/SHA256SUMS"), PACKAGE_RELATIVE,
    )
    write_new_bytes(METHOD_ROOT / "TERMINAL_SHA256SUMS", "".join(
        "%s  %s\n" % (sha256_file(METHOD_ROOT / relative), relative.as_posix())
        for relative in terminal_members
    ).encode("utf-8"))
    print(json.dumps({
        "status": freeze["status"], "campaign_status": freeze["campaign_status"],
        "source_ledger_sha256": freeze["source_ledger_sha256"],
        "method_freeze_sha256": sha256_file(METHOD_ROOT / "method-freeze.json"),
        "package_sha256": sha256_file(package),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

