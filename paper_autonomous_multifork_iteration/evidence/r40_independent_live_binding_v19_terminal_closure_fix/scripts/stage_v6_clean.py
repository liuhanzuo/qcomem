from __future__ import annotations

"""Cross-platform, fail-closed staging for the canonical R39-v6 archive.

The canonical archive was produced on macOS and contains regular AppleDouble
``._*`` members.  BSD tar consumes those members as metadata, whereas GNU tar
and Python's tarfile module materialize them as ordinary files.  This module
defines the only permitted normalization: exclude exact, validated AppleDouble
metadata members and retain every logical member byte-for-byte.
"""

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
import secrets
import shutil
import stat
import struct
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


STATUS = "HOLD_PENDING_FRESH_AUDIT_AND_H20"
V6_ARCHIVE_SHA256 = "306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82"
V19_PACKAGE_NAME = "r40_independent_live_binding_v19_terminal_closure_fix"
STAGE_RECEIPT = ".r40-v19-stage-receipt.json"

EXPECTED_ARCHIVE_MEMBERS = 260
EXPECTED_RETAINED_MEMBERS = 130
EXPECTED_EXCLUDED_MEMBERS = 130
EXPECTED_APPLEDOUBLE_SHA256 = "1cb3b508e9a54815d3d9107032bd897e9cfc84df9b0731e1a59acf2b1b1e0df6"
APPLEDOUBLE_MAGIC = 0x00051607
APPLEDOUBLE_VERSION = 0x00020000
APPLEDOUBLE_HOME_FILESYSTEM = b"Mac OS X        "
APPLEDOUBLE_ENTRIES = ((9, 50, 113), (2, 163, 0))

CLEAN_SCHEMA = "forkaudit-r40-v18-v6-clean-members-v1"
EXCLUSION_SCHEMA = "forkaudit-r40-v18-v6-appledouble-exclusions-v1"
RECEIPT_SCHEMA = "forkaudit-r40-v19-self-contained-stage-receipt-v1"

REQUIRED_SCIENCE_PATHS = (
    "paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/source-code.sha256",
    "paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/dependency-files.sha256",
    "paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/focused-test-fixtures.source.sha256",
    "paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/focused-test-fixtures.archive.sha256",
    "paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh",
    "paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_rank_entrypoint.py",
    "paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_inputs/qcomem_gpu_forkaudit_rr2_formal_20260818w-inputs/code.sha256",
    "paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/gpu/run_qcomem_qwen35_forkaudit_review_revision.py",
    "paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/gpu/qcomem_vllm_paged_multifork_resident.py",
    "paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/results/gpu-qwen35-vllm-paged-fair-v2-20260814c/scientific-artifacts.sha256",
    *(f"paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/results/gpu-qwen35-vllm-paged-fair-v2-20260814c/pg19-gate-shards/pg19-fair-v2-shard-{rank}.json" for rank in range(8)),
)


class StageContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StageContractError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = None
    result["payload_sha256"] = sha256_bytes(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return result


def seal_ok(value: Mapping[str, Any]) -> bool:
    observed = value.get("payload_sha256")
    candidate = dict(value)
    candidate["payload_sha256"] = None
    return is_sha256(observed) and observed == sha256_bytes(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def safe_relative(value: str, *, label: str) -> str:
    require(
        type(value) is str
        and value
        and not any(character in value for character in ("\x00", "\n", "\r")),
        f"{label} contains an empty/control value",
    )
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"{label} escapes staging root")
    normalized = pure.as_posix()
    require(
        normalized == value and value not in {".", ".."} and all(part not in {"", ".", ".."} for part in pure.parts),
        f"{label} is not a strict normalized relative path",
    )
    return normalized


def canonical_file(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    metadata = os.lstat(lexical)
    require(
        stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode) and metadata.st_nlink == 1,
        f"{label} must be an exact single-link regular file",
    )
    require(lexical.resolve(strict=True) == lexical, f"{label} canonical path drift")
    return lexical


def canonical_directory(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    metadata = os.lstat(lexical)
    require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{label} must be an exact directory",
    )
    require(lexical.resolve(strict=True) == lexical, f"{label} canonical path drift")
    return lexical


def read_regular(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "regular file type/link drift")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        require(identity_before == identity_after and len(data) == after.st_size, "regular file changed during read")
        return data, after
    finally:
        os.close(descriptor)


def file_sha256(path: Path) -> str:
    return sha256_bytes(read_regular(path)[0])


def appledouble_companion(path: str) -> str:
    pure = PurePosixPath(path)
    require(pure.name.startswith("._") and len(pure.name) > 2, "excluded member basename is not AppleDouble")
    companion = pure.with_name(pure.name[2:])
    return safe_relative(companion.as_posix(), label="AppleDouble companion")


def validate_appledouble(data: bytes, *, path: str) -> dict[str, Any]:
    require(len(data) == 163, f"AppleDouble metadata byte count drift: {path}")
    require(sha256_bytes(data) == EXPECTED_APPLEDOUBLE_SHA256, f"AppleDouble metadata digest drift: {path}")
    require(len(data) >= 26, f"AppleDouble header truncated: {path}")
    magic, version, filesystem, count = struct.unpack(">II16sH", data[:26])
    require(magic == APPLEDOUBLE_MAGIC, f"AppleDouble magic drift: {path}")
    require(version == APPLEDOUBLE_VERSION, f"AppleDouble version drift: {path}")
    require(filesystem == APPLEDOUBLE_HOME_FILESYSTEM, f"AppleDouble home filesystem drift: {path}")
    require(count == len(APPLEDOUBLE_ENTRIES), f"AppleDouble entry count drift: {path}")
    entries = tuple(struct.unpack(">III", data[26 + index * 12 : 38 + index * 12]) for index in range(count))
    require(entries == APPLEDOUBLE_ENTRIES, f"AppleDouble entry descriptors drift: {path}")
    return {
        "magic": f"0x{magic:08x}",
        "version": f"0x{version:08x}",
        "home_filesystem_ascii": filesystem.decode("ascii"),
        "entry_descriptors": [list(row) for row in entries],
    }


def archive_inventory(archive: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bytes]]:
    archive = canonical_file(archive, label="canonical v6 archive")
    archive_bytes, _ = read_regular(archive)
    require(sha256_bytes(archive_bytes) == V6_ARCHIVE_SHA256, "canonical v6 archive SHA-256 drift")
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    regular_bytes: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as stream:
        members = stream.getmembers()
        require(len(members) == EXPECTED_ARCHIVE_MEMBERS, "canonical v6 archive member count drift")
        seen: set[str] = set()
        for member in members:
            relative = safe_relative(member.name, label="canonical v6 archive member")
            require(relative not in seen, f"duplicate canonical v6 archive member: {relative}")
            seen.add(relative)
            require(member.isfile() or member.isdir(), f"canonical v6 archive link/special member forbidden: {relative}")
            mode = int(member.mode) & 0o7777
            if member.isfile():
                handle = stream.extractfile(member)
                require(handle is not None, f"canonical v6 regular member unreadable: {relative}")
                data = handle.read()
                require(len(data) == member.size, f"canonical v6 regular member short read: {relative}")
                regular_bytes[relative] = data
            else:
                data = b""
                require(member.size == 0, f"canonical v6 directory has payload bytes: {relative}")
            basename = PurePosixPath(relative).name
            if basename.startswith("._"):
                require(member.isfile(), f"AppleDouble exclusion is not a regular member: {relative}")
                proof = validate_appledouble(data, path=relative)
                excluded.append(
                    {
                        "path": relative,
                        "companion_path": appledouble_companion(relative),
                        "type": "regular",
                        "mode": mode,
                        "size": len(data),
                        "sha256": sha256_bytes(data),
                        **proof,
                    }
                )
            else:
                require(
                    relative == "paper_autonomous_multifork_iteration"
                    or relative.startswith("paper_autonomous_multifork_iteration/"),
                    f"retained member escaped exact package root: {relative}",
                )
                retained.append(
                    {
                        "path": relative,
                        "type": "regular" if member.isfile() else "directory",
                        "mode": mode,
                        "size": len(data),
                        "sha256": sha256_bytes(data) if member.isfile() else None,
                    }
                )
    retained.sort(key=lambda row: row["path"])
    excluded.sort(key=lambda row: row["path"])
    require(len(retained) == EXPECTED_RETAINED_MEMBERS, "canonical v6 retained-member count drift")
    require(len(excluded) == EXPECTED_EXCLUDED_MEMBERS, "canonical v6 AppleDouble exclusion count drift")
    retained_by_path = {row["path"]: row for row in retained}
    require(len(retained_by_path) == len(retained), "duplicate retained logical member")
    for row in excluded:
        companion = retained_by_path.get(row["companion_path"])
        require(companion is not None, f"AppleDouble metadata lacks retained logical companion: {row['path']}")
        require(companion["mode"] == row["mode"], f"AppleDouble/companion mode drift: {row['path']}")
    require(all(path in retained_by_path for path in REQUIRED_SCIENCE_PATHS), "required scientific member was excluded/absent")
    require(not any(PurePosixPath(row["path"]).name.startswith("._") for row in retained), "AppleDouble member retained")
    return retained, excluded, regular_bytes


def clean_ledger_value(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    regular = sum(row["type"] == "regular" for row in rows)
    directories = len(rows) - regular
    return seal(
        {
            "schema_version": CLEAN_SCHEMA,
            "status": STATUS,
            "canonical_v6_archive_sha256": V6_ARCHIVE_SHA256,
            "retained_member_count": len(rows),
            "retained_regular_file_count": regular,
            "retained_directory_count": directories,
            "required_science_paths": list(REQUIRED_SCIENCE_PATHS),
            "rows": [dict(row) for row in rows],
            "payload_sha256": None,
        }
    )


def exclusion_ledger_value(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return seal(
        {
            "schema_version": EXCLUSION_SCHEMA,
            "status": STATUS,
            "canonical_v6_archive_sha256": V6_ARCHIVE_SHA256,
            "excluded_member_count": len(rows),
            "exclusion_rule": "basename-starts-with-._-and-exact-AppleDouble-proof",
            "appledouble_magic": "0x00051607",
            "appledouble_version": "0x00020000",
            "appledouble_member_sha256": EXPECTED_APPLEDOUBLE_SHA256,
            "all_exclusions_have_retained_logical_companions": True,
            "scientific_files_excluded": 0,
            "rows": [dict(row) for row in rows],
            "payload_sha256": None,
        }
    )


def write_json_exclusive(path: Path, value: Mapping[str, Any], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)


def load_json_regular(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    path = canonical_file(path, label=label)
    raw, _ = read_regular(path)
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageContractError(f"{label} is not strict JSON") from error
    require(type(value) is dict and seal_ok(value), f"{label} schema/seal drift")
    require(raw == canonical_json_bytes(value), f"{label} canonical JSON byte encoding drift")
    return value, sha256_bytes(raw)


def verified_ledgers(archive: Path, clean_path: Path, exclusion_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bytes], str, str]:
    retained, excluded, data = archive_inventory(archive)
    clean, clean_sha = load_json_regular(clean_path, label="v6 clean-member ledger")
    exclusion, exclusion_sha = load_json_regular(exclusion_path, label="v6 AppleDouble exclusion ledger")
    require(clean == clean_ledger_value(retained), "frozen v6 clean-member ledger differs from canonical archive")
    require(exclusion == exclusion_ledger_value(excluded), "frozen v6 exclusion ledger differs from canonical archive")
    return retained, excluded, data, clean_sha, exclusion_sha


def overlay_inventory(archive: Path, expected_sha256: str) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    require(is_sha256(expected_sha256), "approved v19 overlay SHA-256 invalid")
    archive = canonical_file(archive, label="v19 overlay archive")
    archive_bytes, _ = read_regular(archive)
    require(sha256_bytes(archive_bytes) == expected_sha256, "approved v19 overlay archive mismatch")
    rows: list[dict[str, Any]] = []
    data_by_path: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as stream:
        seen: set[str] = set()
        for member in stream.getmembers():
            relative = safe_relative(member.name, label="v19 overlay member")
            require(relative not in seen, f"duplicate v19 overlay member: {relative}")
            seen.add(relative)
            require(
                relative == V19_PACKAGE_NAME or relative.startswith(V19_PACKAGE_NAME + "/"),
                f"v19 overlay member escaped exact package root: {relative}",
            )
            require(not PurePosixPath(relative).name.startswith("._"), f"v19 overlay contains AppleDouble member: {relative}")
            require(member.isfile() or member.isdir(), f"v19 overlay link/special member forbidden: {relative}")
            if member.isfile():
                handle = stream.extractfile(member)
                require(handle is not None, f"v19 overlay regular member unreadable: {relative}")
                data = handle.read()
                require(len(data) == member.size, f"v19 overlay regular member short read: {relative}")
                data_by_path[relative] = data
            else:
                data = b""
                require(member.size == 0, f"v19 overlay directory has payload bytes: {relative}")
            rows.append(
                {
                    "path": relative,
                    "type": "regular" if member.isfile() else "directory",
                    "mode": int(member.mode) & 0o7777,
                    "size": len(data),
                    "sha256": sha256_bytes(data) if member.isfile() else None,
                }
            )
    rows.sort(key=lambda row: row["path"])
    require(rows and len({row["path"] for row in rows}) == len(rows), "v19 overlay inventory empty/duplicate")
    return rows, data_by_path


def overlay_stage_rows(rows: Sequence[Mapping[str, Any]], retained_paths: set[str]) -> list[dict[str, Any]]:
    prefix = PurePosixPath("paper_autonomous_multifork_iteration/evidence")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        mapped = (prefix / row["path"]).as_posix()
        require(mapped not in retained_paths, f"v19 overlay collides with canonical v6 member: {mapped}")
        result[mapped] = {**dict(row), "path": mapped}
        for parent in PurePosixPath(mapped).parents:
            if parent == PurePosixPath("."):
                break
            relative = parent.as_posix()
            if relative in retained_paths:
                continue
            existing = result.get(relative)
            expected = {"path": relative, "type": "directory", "mode": 0o755, "size": 0, "sha256": None}
            require(existing is None or existing == expected, f"v19 overlay implied-directory drift: {relative}")
            result[relative] = expected
    return sorted(result.values(), key=lambda row: row["path"])


def ensure_directory(root: Path, relative: str, *, final_mode: int = 0o755) -> Path:
    current = root
    for part in PurePosixPath(safe_relative(relative, label="stage directory")).parts:
        current = current / part
        if os.path.lexists(current):
            metadata = os.lstat(current)
            require(stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), f"stage parent is not an exact directory: {current}")
        else:
            current.mkdir(mode=0o755)
    os.chmod(current, final_mode)
    return current


def write_stage_file(root: Path, relative: str, data: bytes, *, mode: int) -> None:
    pure = PurePosixPath(safe_relative(relative, label="stage regular file"))
    if pure.parent != PurePosixPath("."):
        ensure_directory(root, pure.parent.as_posix())
    path = root.joinpath(*pure.parts)
    require(not os.path.lexists(path), f"stage member overwrite/collision: {relative}")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)


def extract_rows(root: Path, rows: Sequence[Mapping[str, Any]], data_by_path: Mapping[str, bytes], *, source_prefix: str = "") -> None:
    for row in sorted(rows, key=lambda item: (item["path"].count("/"), item["path"])):
        stage_relative = row["path"]
        source_relative = stage_relative[len(source_prefix) :] if source_prefix else stage_relative
        if source_prefix:
            require(stage_relative.startswith(source_prefix), "overlay source-prefix projection drift")
        if row["type"] == "directory":
            ensure_directory(root, stage_relative, final_mode=int(row["mode"]))
        else:
            require(source_relative in data_by_path, f"stage source bytes absent: {source_relative}")
            write_stage_file(root, stage_relative, data_by_path[source_relative], mode=int(row["mode"]))


def lexical_stage_tree(root: Path) -> list[dict[str, Any]]:
    root = canonical_directory(root, label="stage root")
    rows: list[dict[str, Any]] = []

    def walk(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                path = directory / entry.name
                relative = path.relative_to(root).as_posix()
                safe_relative(relative, label="observed stage member")
                require(not PurePosixPath(relative).name.startswith("._"), f"AppleDouble path remains in staged tree: {relative}")
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise StageContractError(f"staged tree symlink forbidden: {relative}")
                if stat.S_ISDIR(metadata.st_mode):
                    rows.append({"path": relative, "type": "directory", "mode": stat.S_IMODE(metadata.st_mode), "size": 0, "sha256": None})
                    walk(path)
                elif stat.S_ISREG(metadata.st_mode):
                    data, exact = read_regular(path)
                    require((metadata.st_dev, metadata.st_ino) == (exact.st_dev, exact.st_ino), f"staged file identity changed: {relative}")
                    rows.append({"path": relative, "type": "regular", "mode": stat.S_IMODE(exact.st_mode), "size": len(data), "sha256": sha256_bytes(data)})
                else:
                    raise StageContractError(f"staged tree special node forbidden: {relative}")

    walk(root)
    rows.sort(key=lambda row: row["path"])
    return rows


RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "stage_root",
    "canonical_v6_archive_sha256",
    "v19_overlay_archive_sha256",
    "clean_member_ledger_sha256",
    "appledouble_exclusion_ledger_sha256",
    "canonical_member_count",
    "retained_member_count",
    "excluded_appledouble_member_count",
    "overlay_archive_member_count",
    "final_stage_member_count",
    "final_regular_file_count",
    "final_directory_count",
    "zero_appledouble_paths",
    "excluded_only_exact_appledouble_metadata",
    "scientific_files_excluded",
    "v19_source_ledger_sha256",
    "v19_current_payload_manifest_sha256",
    "payload_sha256",
}


def expected_receipt(
    *,
    overlay_sha256: str,
    clean_sha256: str,
    exclusion_sha256: str,
    overlay_archive_member_count: int,
    final_rows_without_receipt: Sequence[Mapping[str, Any]],
    v19_source_ledger_sha256: str,
    v19_payload_manifest_sha256: str,
) -> dict[str, Any]:
    regular_without = sum(row["type"] == "regular" for row in final_rows_without_receipt)
    directory_count = len(final_rows_without_receipt) - regular_without
    return seal(
        {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "stage_root": ".",
            "canonical_v6_archive_sha256": V6_ARCHIVE_SHA256,
            "v19_overlay_archive_sha256": overlay_sha256,
            "clean_member_ledger_sha256": clean_sha256,
            "appledouble_exclusion_ledger_sha256": exclusion_sha256,
            "canonical_member_count": EXPECTED_ARCHIVE_MEMBERS,
            "retained_member_count": EXPECTED_RETAINED_MEMBERS,
            "excluded_appledouble_member_count": EXPECTED_EXCLUDED_MEMBERS,
            "overlay_archive_member_count": overlay_archive_member_count,
            "final_stage_member_count": len(final_rows_without_receipt) + 1,
            "final_regular_file_count": regular_without + 1,
            "final_directory_count": directory_count,
            "zero_appledouble_paths": True,
            "excluded_only_exact_appledouble_metadata": True,
            "scientific_files_excluded": 0,
            "v19_source_ledger_sha256": v19_source_ledger_sha256,
            "v19_current_payload_manifest_sha256": v19_payload_manifest_sha256,
            "payload_sha256": None,
        }
    )


def validate_receipt(value: object) -> dict[str, Any]:
    require(type(value) is dict and set(value) == RECEIPT_FIELDS, "stage receipt exact schema drift")
    value = dict(value)
    require(value["schema_version"] == RECEIPT_SCHEMA and value["status"] == STATUS, "stage receipt identity drift")
    require(seal_ok(value), "stage receipt seal drift")
    return value


def verify_stage(
    *,
    stage_root: Path,
    v6_archive: Path,
    overlay_archive: Path,
    clean_ledger: Path,
    exclusion_ledger: Path,
    expected_v6_sha256: str,
    expected_overlay_sha256: str,
) -> dict[str, Any]:
    require(expected_v6_sha256 == V6_ARCHIVE_SHA256, "operator-approved v6 archive SHA-256 drift")
    require(is_sha256(expected_overlay_sha256), "operator-approved v19 overlay SHA-256 invalid")
    retained, _excluded, _v6_data, clean_sha, exclusion_sha = verified_ledgers(v6_archive, clean_ledger, exclusion_ledger)
    overlay_rows, _overlay_data = overlay_inventory(overlay_archive, expected_overlay_sha256)
    retained_paths = {row["path"] for row in retained}
    mapped_overlay = overlay_stage_rows(overlay_rows, retained_paths)
    expected_without_receipt = sorted([*retained, *mapped_overlay], key=lambda row: row["path"])
    require(len({row["path"] for row in expected_without_receipt}) == len(expected_without_receipt), "final stage expected path collision")
    root = canonical_directory(stage_root, label="stage root")
    receipt_path = root / STAGE_RECEIPT
    receipt_value, _ = load_json_regular(receipt_path, label="stage receipt")
    receipt = validate_receipt(receipt_value)
    source_ledger = root / f"paper_autonomous_multifork_iteration/evidence/{V19_PACKAGE_NAME}/source-code.sha256"
    payload_manifest = root / f"paper_autonomous_multifork_iteration/evidence/{V19_PACKAGE_NAME}/v19-current-payload.sha256"
    expected_value = expected_receipt(
        overlay_sha256=expected_overlay_sha256,
        clean_sha256=clean_sha,
        exclusion_sha256=exclusion_sha,
        overlay_archive_member_count=len(overlay_rows),
        final_rows_without_receipt=expected_without_receipt,
        v19_source_ledger_sha256=file_sha256(source_ledger),
        v19_payload_manifest_sha256=file_sha256(payload_manifest),
    )
    require(receipt == expected_value, "stage receipt differs from exact archive/ledger/tree contract")
    observed = lexical_stage_tree(root)
    observed_by_path = {row["path"]: row for row in observed}
    expected_by_path = {row["path"]: dict(row) for row in expected_without_receipt}
    receipt_bytes, receipt_metadata = read_regular(receipt_path)
    require(stat.S_IMODE(receipt_metadata.st_mode) == 0o600, "stage receipt mode drift")
    expected_by_path[STAGE_RECEIPT] = {
        "path": STAGE_RECEIPT,
        "type": "regular",
        "mode": 0o600,
        "size": len(receipt_bytes),
        "sha256": sha256_bytes(receipt_bytes),
    }
    require(set(observed_by_path) == set(expected_by_path), "final stage exact path whitelist drift")
    for relative, expected in expected_by_path.items():
        require(observed_by_path[relative] == expected, f"final stage byte/mode/type drift: {relative}")
    require(not any(PurePosixPath(row["path"]).name.startswith("._") for row in observed), "final stage contains AppleDouble path")
    return receipt


def normalize_new_output(path: Path) -> tuple[Path, Path]:
    raw = Path(path)
    require(".." not in raw.parts, "stage output contains forbidden dotdot component")
    lexical = Path(os.path.abspath(os.fspath(raw)))
    parent = canonical_directory(lexical.parent, label="stage output parent")
    require(lexical.parent == parent, "stage output parent canonical drift")
    require(not os.path.lexists(lexical), "stage output already exists or is a special node")
    return lexical, parent


def rename_directory_noreplace(source: Path, destination: Path) -> None:
    require(source.parent == destination.parent, "atomic stage publication requires one exact parent")
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin" and hasattr(library, "renamex_np"):
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source_bytes, destination_bytes, 0x00000004)
    else:
        raise StageContractError("atomic no-replace directory publication unavailable on this platform")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise StageContractError("stage output appeared before atomic no-replace publication")
        raise OSError(error_number, os.strerror(error_number), os.fspath(destination))


def prepare_stage(
    *,
    output_root: Path,
    v6_archive: Path,
    overlay_archive: Path,
    clean_ledger: Path,
    exclusion_ledger: Path,
    expected_v6_sha256: str,
    expected_overlay_sha256: str,
) -> dict[str, Any]:
    require(expected_v6_sha256 == V6_ARCHIVE_SHA256, "operator-approved v6 archive SHA-256 drift")
    output, parent = normalize_new_output(output_root)
    retained, _excluded, v6_data, clean_sha, exclusion_sha = verified_ledgers(v6_archive, clean_ledger, exclusion_ledger)
    overlay_rows, overlay_data = overlay_inventory(overlay_archive, expected_overlay_sha256)
    mapped_overlay = overlay_stage_rows(overlay_rows, {row["path"] for row in retained})
    expected_without_receipt = sorted([*retained, *mapped_overlay], key=lambda row: row["path"])
    temporary = parent / f".{output.name}.staging-{os.getpid()}-{secrets.token_hex(8)}"
    require(not os.path.lexists(temporary), "private staging temporary path collision")
    temporary.mkdir(mode=0o700)
    try:
        extract_rows(temporary, retained, v6_data)
        prefix = "paper_autonomous_multifork_iteration/evidence/"
        extract_rows(temporary, mapped_overlay, overlay_data, source_prefix=prefix)
        source_ledger = temporary / f"paper_autonomous_multifork_iteration/evidence/{V19_PACKAGE_NAME}/source-code.sha256"
        payload_manifest = temporary / f"paper_autonomous_multifork_iteration/evidence/{V19_PACKAGE_NAME}/v19-current-payload.sha256"
        receipt = expected_receipt(
            overlay_sha256=expected_overlay_sha256,
            clean_sha256=clean_sha,
            exclusion_sha256=exclusion_sha,
            overlay_archive_member_count=len(overlay_rows),
            final_rows_without_receipt=expected_without_receipt,
            v19_source_ledger_sha256=file_sha256(source_ledger),
            v19_payload_manifest_sha256=file_sha256(payload_manifest),
        )
        write_json_exclusive(temporary / STAGE_RECEIPT, receipt, mode=0o600)
        verify_stage(
            stage_root=temporary,
            v6_archive=v6_archive,
            overlay_archive=overlay_archive,
            clean_ledger=clean_ledger,
            exclusion_ledger=exclusion_ledger,
            expected_v6_sha256=expected_v6_sha256,
            expected_overlay_sha256=expected_overlay_sha256,
        )
        rename_directory_noreplace(temporary, output)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return verify_stage(
        stage_root=output,
        v6_archive=v6_archive,
        overlay_archive=overlay_archive,
        clean_ledger=clean_ledger,
        exclusion_ledger=exclusion_ledger,
        expected_v6_sha256=expected_v6_sha256,
        expected_overlay_sha256=expected_overlay_sha256,
    )


def freeze_ledgers(archive: Path, clean_output: Path, exclusion_output: Path) -> None:
    retained, excluded, _ = archive_inventory(archive)
    write_json_exclusive(clean_output, clean_ledger_value(retained))
    write_json_exclusive(exclusion_output, exclusion_ledger_value(excluded))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-ledgers")
    freeze.add_argument("--v6-archive", type=Path, required=True)
    freeze.add_argument("--clean-output", type=Path, required=True)
    freeze.add_argument("--exclusion-output", type=Path, required=True)
    for name in ("prepare", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--v6-archive", type=Path, required=True)
        command.add_argument("--overlay-archive", type=Path, required=True)
        command.add_argument("--clean-ledger", type=Path, required=True)
        command.add_argument("--exclusion-ledger", type=Path, required=True)
        command.add_argument("--expected-v6-sha256", required=True)
        command.add_argument("--expected-overlay-sha256", required=True)
        command.add_argument("--output-root" if name == "prepare" else "--stage-root", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "freeze-ledgers":
        freeze_ledgers(args.v6_archive, args.clean_output, args.exclusion_output)
        return 0
    keyword = {
        "v6_archive": args.v6_archive,
        "overlay_archive": args.overlay_archive,
        "clean_ledger": args.clean_ledger,
        "exclusion_ledger": args.exclusion_ledger,
        "expected_v6_sha256": args.expected_v6_sha256,
        "expected_overlay_sha256": args.expected_overlay_sha256,
    }
    value = prepare_stage(output_root=args.output_root, **keyword) if args.command == "prepare" else verify_stage(stage_root=args.stage_root, **keyword)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
