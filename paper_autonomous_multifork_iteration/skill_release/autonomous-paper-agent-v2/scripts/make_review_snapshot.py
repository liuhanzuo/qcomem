#!/usr/bin/env python3
"""Create separate PDF-only and artifact-aware blind-review snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                yield child
        return
    raise FileNotFoundError(path)


def copy_input(source: Path, destination: Path, label: str) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    if source.is_file():
        target = destination / label / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "snapshot_path": target.relative_to(destination).as_posix(),
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )
        return records

    root = source.resolve()
    for item in iter_files(source):
        relative = item.resolve().relative_to(root)
        target = destination / label / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        records.append(
            {
                "snapshot_path": target.relative_to(destination).as_posix(),
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )
    return records


def snapshot_digest(records: list[dict[str, str | int]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: str(row["snapshot_path"])):
        digest.update(str(record["snapshot_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, help="Zero-based review round number.")
    parser.add_argument("--paper", type=Path, required=True, help="Compiled manuscript PDF.")
    parser.add_argument(
        "--pdf-only-include",
        action="append",
        required=True,
        type=Path,
        help="Venue instruction or rubric visible to both reviewer modes. Repeat as needed.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        type=Path,
        help="Anonymous repository/evidence input visible only to artifact-aware reviewers.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("review"))
    parser.add_argument("--force", action="store_true", help="Replace an existing round snapshot.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.round < 0:
        raise ValueError("--round must be non-negative")
    if not args.paper.is_file():
        raise FileNotFoundError(f"Paper PDF not found: {args.paper}")

    round_dir = args.output_root / f"round_{args.round:02d}"
    destination = round_dir / "submission"
    if destination.exists():
        if not args.force:
            raise FileExistsError(f"Snapshot already exists: {destination}; use --force to replace it")
        shutil.rmtree(destination)
    pdf_only = destination / "pdf_only"
    artifact_aware = destination / "pdf_plus_repository"
    pdf_only.mkdir(parents=True, exist_ok=False)
    artifact_aware.mkdir(parents=True, exist_ok=False)

    pdf_records = copy_input(args.paper, pdf_only, "manuscript")
    artifact_records = copy_input(args.paper, artifact_aware, "manuscript")
    paper_sha256 = sha256_file(args.paper)

    used_pdf_labels: set[str] = {"manuscript"}
    for index, source in enumerate(args.pdf_only_include, start=1):
        if not source.exists():
            raise FileNotFoundError(f"PDF-only included path not found: {source}")
        base = source.name or f"rubric_{index}"
        label = base
        suffix = 2
        while label in used_pdf_labels:
            label = f"{base}_{suffix}"
            suffix += 1
        used_pdf_labels.add(label)
        pdf_records.extend(copy_input(source, pdf_only, label))
        artifact_records.extend(copy_input(source, artifact_aware, label))

    used_artifact_labels: set[str] = set(used_pdf_labels)
    for index, source in enumerate(args.include, start=1):
        if not source.exists():
            raise FileNotFoundError(f"Included path not found: {source}")
        base = source.name or f"input_{index}"
        label = base
        suffix = 2
        while label in used_artifact_labels:
            label = f"{base}_{suffix}"
            suffix += 1
        used_artifact_labels.add(label)
        artifact_records.extend(copy_input(source, artifact_aware, label))

    def write_view_manifest(
        view: Path,
        access_mode: str,
        records: list[dict[str, str | int]],
    ) -> dict[str, str | int]:
        manifest = {
            "schema_version": "2.0.0",
            "round": args.round,
            "access_mode": access_mode,
            "snapshot_sha256": snapshot_digest(records),
            "paper_sha256": paper_sha256,
            "files": sorted(records, key=lambda row: str(row["snapshot_path"])),
        }
        manifest_path = view / "MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {
            "access_mode": access_mode,
            "snapshot_sha256": str(manifest["snapshot_sha256"]),
            "manifest_sha256": sha256_file(manifest_path),
            "file_count": len(records),
        }

    pdf_view = write_view_manifest(pdf_only, "pdf_only", pdf_records)
    artifact_view = write_view_manifest(
        artifact_aware, "pdf_plus_repository", artifact_records
    )
    if sha256_file(pdf_only / "manuscript" / args.paper.name) != sha256_file(
        artifact_aware / "manuscript" / args.paper.name
    ):
        raise RuntimeError("Reviewer views contain different manuscript bytes")

    manifest = {
        "schema_version": "2.0.0",
        "round": args.round,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_sha256": paper_sha256,
        "default_panel_access_mix": {
            "pdf_only": 3,
            "pdf_plus_repository": 2,
        },
        "views": [pdf_view, artifact_view],
    }
    manifest_path = destination / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"snapshot_dir": destination.as_posix(), **manifest}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
