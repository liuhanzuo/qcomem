#!/usr/bin/env python3
"""Strict path, evidence-ID, manifest, and Python-symbol audit for a snapshot."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any


EVIDENCE_ID_RE = re.compile(r"^E-[A-Z0-9][A-Z0-9_-]*$")
MANUSCRIPT_LABEL_RE = re.compile(r"\[label:([A-Za-z0-9:_-]+)\]")


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_path(root: Path, relative: str) -> Path:
    rel = Path(relative)
    require(relative and not rel.is_absolute() and ".." not in rel.parts, f"unsafe path: {relative!r}")
    path = root / rel
    require(path.exists(), f"missing path: {relative}")
    return path


def verify_snapshot_manifest(root: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json(root / "MANIFEST.json")
    rows = manifest.get("files")
    require(isinstance(rows, list), "snapshot MANIFEST.json has no files list")
    index: dict[str, dict[str, Any]] = {}
    identity = hashlib.sha256()
    total_bytes = 0
    for row in rows:
        relative = row.get("path")
        require(isinstance(relative, str) and relative not in index, "duplicate/invalid snapshot path")
        path = safe_path(root, relative)
        require(path.is_file(), f"manifest member is not a file: {relative}")
        observed_size = path.stat().st_size
        observed_sha = sha256(path)
        require(row.get("size_bytes") == observed_size, f"snapshot size mismatch: {relative}")
        require(row.get("sha256") == observed_sha, f"snapshot SHA mismatch: {relative}")
        index[relative] = row
        total_bytes += observed_size
        identity.update(relative.encode("utf-8"))
        identity.update(b"\0")
        identity.update(observed_sha.encode("ascii"))
        identity.update(b"\n")
    require(manifest.get("file_count") == len(rows), "snapshot file_count mismatch")
    require(manifest.get("total_bytes") == total_bytes, "snapshot total_bytes mismatch")
    require(manifest.get("snapshot_sha256") == identity.hexdigest(), "snapshot identity mismatch")
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        require(not path.is_symlink(), f"snapshot contains a symlink: {relative}")
        require(path.is_file() or path.is_dir(), f"snapshot contains a special entry: {relative}")
        if path.is_file():
            actual_files.add(relative)
    expected_files = set(index) | {"MANIFEST.json"}
    require(
        actual_files == expected_files,
        "snapshot exact file-set mismatch: "
        f"missing={sorted(expected_files - actual_files)}, "
        f"extra={sorted(actual_files - expected_files)}",
    )
    return index


def require_manifest_bound(path: Path, root: Path, manifest_index: dict[str, Any]) -> None:
    relative = path.relative_to(root).as_posix()
    require(relative in manifest_index, f"path is not bound by snapshot manifest: {relative}")


def load_registry(root: Path, manifest_index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = safe_path(root, "evidence/experiment_registry.json")
    require_manifest_bound(path, root, manifest_index)
    registry = load_json(path)
    rows = registry.get("experiments")
    require(isinstance(rows, list) and rows, "global experiment registry is empty")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        evidence_id = row.get("evidence_id")
        require(isinstance(evidence_id, str) and EVIDENCE_ID_RE.fullmatch(evidence_id), f"invalid evidence ID: {evidence_id!r}")
        require(evidence_id not in by_id, f"duplicate evidence ID: {evidence_id}")
        paths = row.get("artifact_paths")
        require(isinstance(paths, list) and paths, f"{evidence_id} has no artifact_paths")
        for relative in paths:
            artifact = safe_path(root, relative)
            require(artifact.is_file(), f"registry artifact is not a file: {relative}")
            require_manifest_bound(artifact, root, manifest_index)
        for artifact_set in row.get("artifact_sets", []):
            pattern = artifact_set.get("glob")
            expected_count = artifact_set.get("expected_count")
            require(isinstance(pattern, str) and ".." not in Path(pattern).parts, f"unsafe artifact glob in {evidence_id}")
            matches = sorted(root.glob(pattern))
            require(len(matches) == expected_count, f"artifact-set count mismatch for {evidence_id}: {pattern}")
            for artifact in matches:
                require(artifact.is_file(), f"artifact-set member is not a file: {artifact}")
                require_manifest_bound(artifact, root, manifest_index)
        command = row.get("one_command_replay")
        if command:
            tokens = shlex.split(command)
            require(len(tokens) >= 4 and tokens[0] == "cd" and tokens[2] == "&&", f"unsupported replay command form: {command}")
            workdir = safe_path(root, tokens[1])
            require(workdir.is_dir(), f"replay workdir is not a directory: {tokens[1]}")
            entry = safe_path(workdir, tokens[3][2:] if tokens[3].startswith("./") else tokens[3])
            require(entry.is_file(), f"replay entry is not a file: {command}")
            require_manifest_bound(entry, root, manifest_index)
        by_id[evidence_id] = row
    return by_id


def claim_evidence_ids(root: Path, manifest_index: dict[str, Any]) -> set[str]:
    path = safe_path(root, "evidence/claim_evidence_map.tsv")
    require_manifest_bound(path, root, manifest_index)
    found: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames is not None and "evidence_ids" in reader.fieldnames, "claim map lacks evidence_ids")
        for row in reader:
            for evidence_id in row["evidence_ids"].split(";"):
                evidence_id = evidence_id.strip()
                if not evidence_id:
                    continue
                require(EVIDENCE_ID_RE.fullmatch(evidence_id) is not None, f"invalid claim evidence ID: {evidence_id}")
                found.add(evidence_id)
    return found


def python_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.add(f"{node.name}.{member.name}")
    return symbols


def validate_methods(root: Path, manifest_index: dict[str, Any]) -> tuple[int, int]:
    path = safe_path(root, "evidence/method_provenance.tsv")
    require_manifest_bound(path, root, manifest_index)
    rows = 0
    checked_symbols = 0
    seen: set[str] = set()
    symbol_cache: dict[Path, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"method_id", "source_path", "symbol_or_lines"}
        require(reader.fieldnames is not None and required.issubset(reader.fieldnames), "method map schema drift")
        for row in reader:
            method_id = row["method_id"]
            require(method_id and method_id not in seen, f"duplicate/empty method ID: {method_id!r}")
            seen.add(method_id)
            source = safe_path(root, row["source_path"])
            require(source.is_file(), f"method source is not a file: {row['source_path']}")
            require_manifest_bound(source, root, manifest_index)
            anchors = [part.strip() for part in row["symbol_or_lines"].split(";") if part.strip()]
            require(anchors, f"method row has no symbol/anchor: {method_id}")
            if source.suffix == ".py":
                symbols = symbol_cache.setdefault(source, python_symbols(source))
                for anchor in anchors:
                    require(anchor in symbols, f"unresolved Python symbol for {method_id}: {anchor} in {row['source_path']}")
                    checked_symbols += 1
            rows += 1
    return rows, checked_symbols


def validate_status_artifact(root: Path, manifest_index: dict[str, Any]) -> list[str]:
    path = safe_path(root, "evidence/seven_target_status.json")
    require_manifest_bound(path, root, manifest_index)
    value = load_json(path)
    targets = value.get("targets")
    require(isinstance(targets, list) and len(targets) == 7, "seven-target artifact does not have seven rows")
    statuses = []
    for index, row in enumerate(targets, start=1):
        require(row.get("target_index") == index, "target index/order drift")
        status = row.get("status")
        require(status in {"full", "partial", "open"}, f"invalid target status: {status}")
        require(isinstance(row.get("decisive_predicate"), str) and row["decisive_predicate"], "missing decisive predicate")
        require(isinstance(row.get("missing_records"), list), "missing_records must be explicit")
        for relative in row.get("evidence", []):
            evidence = safe_path(root, relative)
            if evidence.is_file():
                require_manifest_bound(evidence, root, manifest_index)
            else:
                bound_children = [name for name in manifest_index if name.startswith(relative.rstrip("/") + "/")]
                require(bound_children, f"status evidence directory has no manifest-bound members: {relative}")
        statuses.append(status)
    require(value.get("status_vector") == statuses, "status_vector disagrees with target rows")
    expected_overall = "open" if "open" in statuses else ("partial" if "partial" in statuses else "full")
    require(value.get("overall_status") == expected_overall, "overall target status is inconsistent")
    return statuses


def validate_manuscript_location_labels(
    root: Path, manifest_index: dict[str, Any]
) -> int:
    manuscript = safe_path(root, "source/main.tex")
    require_manifest_bound(manuscript, root, manifest_index)
    reachable = {manuscript}
    pending = [manuscript]
    input_re = re.compile(r"\\input\{([^}]+)\}")
    while pending:
        source = pending.pop()
        for relative in input_re.findall(source.read_text(encoding="utf-8")):
            target = (source.parent / relative).resolve()
            if target.suffix == "":
                target = target.with_suffix(".tex")
            require(
                root in target.parents and target.is_file(),
                f"unresolved manuscript input: {relative} from {source}",
            )
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    defined: set[str] = set()
    for source in sorted(reachable):
        require_manifest_bound(source, root, manifest_index)
        defined.update(
            re.findall(
                r"\\label\{([A-Za-z0-9:_-]+)\}",
                source.read_text(encoding="utf-8"),
            )
        )
    checked = 0
    checked_rows = 0
    unlocated_ids: set[str] = set()
    for relative in (
        "evidence/claim_evidence_map.tsv",
        "evidence/method_provenance.tsv",
    ):
        path = safe_path(root, relative)
        require_manifest_bound(path, root, manifest_index)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require(
                reader.fieldnames is not None
                and "manuscript_locations" in reader.fieldnames,
                f"{relative} lacks manuscript_locations",
            )
            for row in reader:
                location = row["manuscript_locations"].strip()
                if location == "N/A":
                    identifier = (row.get("claim_id") or row.get("method_id") or "").strip()
                    require(identifier, f"N/A row has no stable ID in {relative}: {row}")
                    unlocated_ids.add(identifier)
                    continue
                labels = MANUSCRIPT_LABEL_RE.findall(location)
                require(
                    labels,
                    f"unchecked manuscript location row in {relative}: {row}",
                )
                require(
                    len(labels) == len(set(labels)),
                    f"duplicate manuscript label in {relative}: {labels}",
                )
                for label in labels:
                    require(
                        label in defined,
                        f"unresolved manuscript label in {relative}: {label}",
                    )
                    checked += 1
                checked_rows += 1
    expected_unlocated = {
        "C-UNSUPPORTED-SPEED",
        "C-UNSUPPORTED-GENERAL",
        "C-UNSUPPORTED-SEMANTIC",
        "C-UNSUPPORTED-GDN",
        "C-UNSUPPORTED-SCHEDULER-GENERAL",
        "M-R6-TRANSFER-STORAGE",
        "M-R6-TRANSFER-EXECUTION",
        "M-R6-TRANSFER-AGGREGATE",
    }
    require(
        unlocated_ids == expected_unlocated,
        f"unexpected N/A claim/method rows: {sorted(unlocated_ids)}",
    )
    require(checked_rows >= 44, "too few location-bound claim/method rows")
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.snapshot_root).resolve()
    require((root / "MANIFEST.json").is_file(), "run against a frozen snapshot root containing MANIFEST.json")
    manifest_index = verify_snapshot_manifest(root)
    registry = load_registry(root, manifest_index)
    claim_ids = claim_evidence_ids(root, manifest_index)
    missing_ids = sorted(claim_ids - set(registry))
    require(not missing_ids, f"claim evidence IDs absent from registry: {missing_ids}")
    method_rows, checked_symbols = validate_methods(root, manifest_index)
    statuses = validate_status_artifact(root, manifest_index)
    location_labels = validate_manuscript_location_labels(root, manifest_index)
    manifest = load_json(root / "MANIFEST.json")
    result = {
        "status": "PASS",
        "schema_version": "forkaudit-snapshot-provenance-audit-v1",
        "snapshot_sha256": manifest["snapshot_sha256"],
        "manifest_files_verified": len(manifest_index),
        "registered_evidence_ids": len(registry),
        "claim_evidence_ids_resolved": len(claim_ids),
        "method_rows_resolved": method_rows,
        "python_symbols_resolved": checked_symbols,
        "manuscript_location_labels_resolved": location_labels,
        "seven_target_status_vector": statuses,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
