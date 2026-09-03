#!/usr/bin/env python3
"""Create reviewer-safe round snapshots without exposing review history.

The Round-1 submission directory supplies only the already-vetted artifact and
rubric inventory.  Current manuscript, evidence, literature, protocol, and
audit files replace their old counterparts before a new byte manifest is made.
"""
from __future__ import annotations

import datetime as dt
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = (
    "a11-correlated-majority-vote",
    "a2-subgroup-mix-ranking",
    "a2-erase-late-absorb-early",
)
SOURCE_SUFFIXES = {".tex", ".bib", ".sty", ".bst", ".png", ".pdf", ".py", ".json", ".tsv", ".txt"}
IDENTITY_MARKERS = (b"/Users/liuhanzuo", b"/newcpfs/user/qixuan", b"THUQiXuan")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_current_source(project: Path, stage: Path) -> None:
    target = stage / "source"
    shutil.rmtree(target)
    target.mkdir()
    manuscript = project / "manuscript"
    for path in sorted(manuscript.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES or path.name == "paper.pdf":
            continue
        rel = path.relative_to(manuscript)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)


def replace_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def freeze(name: str, round_number: int) -> tuple[str, int]:
    project = ROOT / name
    previous = max(1, round_number - 1)
    seed = project / f"review/round_{previous:02d}/submission"
    if not seed.is_dir():
        seed = project / "review/round_01/submission"
    destination = project / f"review/round_{round_number:02d}/submission"
    if destination.parent.exists():
        raise SystemExit(f"refusing to overwrite existing round directory: {destination.parent}")

    temp_root = Path("/private/tmp") if Path("/private/tmp").is_dir() else Path("/tmp")
    stage = Path(tempfile.mkdtemp(prefix=f"{name}-r{round_number}-freeze.", dir=temp_root))
    try:
        shutil.copytree(seed, stage, dirs_exist_ok=True)
        (stage / "MANIFEST.json").unlink(missing_ok=True)

        copy_current_source(project, stage)
        shutil.copy2(project / "manuscript/paper.pdf", stage / "manuscript/paper.pdf")
        replace_tree(project / "evidence", stage / "evidence")
        replace_tree(project / "literature", stage / "literature")

        audit = stage / "audit"
        if audit.exists():
            shutil.rmtree(audit)
        audit.mkdir()
        shutil.copy2(project / "build/build_record.json", audit / "build_record.json")
        r2_audit = project / "review/pre_review_audit_round_02.json"
        base_audit = project / "review/pre_review_audit.json"
        if r2_audit.is_file():
            shutil.copy2(r2_audit, audit / "pre_review_audit_round_02.json")
        elif base_audit.is_file():
            shutil.copy2(base_audit, audit / "pre_review_audit.json")
        scope_audit = project / "review/revision_02_t2_scope_audit.md"
        if scope_audit.is_file():
            shutil.copy2(scope_audit, audit / scope_audit.name)

        protocol = project / "experiments/decisive_experiment_protocol.md"
        if protocol.is_file():
            (stage / "protocol").mkdir(exist_ok=True)
            shutil.copy2(protocol, stage / "protocol/decisive_experiment_protocol.md")

        # Reviewer-runnable inputs for the two lightweight, read-only audits.
        if name == "a11-correlated-majority-vote":
            rel = Path("results/tv_conservation_r484_result.json")
            src = project / "remote_snapshot" / rel
            dst = stage / "remote_snapshot" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        if name == "a2-subgroup-mix-ranking":
            rel = Path("results/SUBGMIX_M25_PAIRED_R1885_5SEED.json")
            src = project / "remote_snapshot" / rel
            dst = stage / "remote_snapshot" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        note = stage / "SNAPSHOT_SCOPE.md"
        note.write_text(
            f"# Frozen Round-{round_number} reviewer snapshot\n\n"
            "Review only this directory under the bundled protocol and rubric. "
            "It contains no prior reviews, author response, target score, or revision plan.\n\n"
            "The supplied audits are evidence records, not substitutes for independent "
            "technical judgment.  Missing raw data or environment material must be scored "
            "as a reproducibility boundary rather than silently inferred.\n",
            encoding="utf-8",
        )
        verifier = stage / "VERIFY_SNAPSHOT.py"
        verifier.write_text(
            '''#!/usr/bin/env python3
"""Verify every frozen member and the path-ordered snapshot root."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
lines = []
for row in sorted(MANIFEST["files"], key=lambda item: item["snapshot_path"].encode("utf-8")):
    path = HERE / row["snapshot_path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != row["sha256"] or path.stat().st_size != row["size_bytes"]:
        raise SystemExit(f"FAIL {row['snapshot_path']}")
    lines.append(f"{actual}  {row['snapshot_path']}\\n")
root = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
if root != MANIFEST["snapshot_sha256"]:
    raise SystemExit(f"FAIL root {root} != {MANIFEST['snapshot_sha256']}")
print(f"PASS {len(lines)} files root={root}")
''',
            encoding="utf-8",
        )

        offenders = []
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            data = path.read_bytes()
            for marker in IDENTITY_MARKERS:
                if marker in data:
                    offenders.append(f"{path.relative_to(stage)}:{marker.decode()}")
        if offenders:
            raise SystemExit("identity markers in staged snapshot: " + ", ".join(offenders))

        rows = []
        member_paths = [p for p in stage.rglob("*") if p.is_file() and p.name != "MANIFEST.json"]
        for path in sorted(member_paths, key=lambda p: p.relative_to(stage).as_posix().encode("utf-8")):
            rel = path.relative_to(stage).as_posix()
            rows.append(
                {
                    "snapshot_path": rel,
                    "source_path": str(path),
                    "sha256": digest(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        canonical = "".join(f"{row['sha256']}  {row['snapshot_path']}\n" for row in rows)
        snapshot_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        manifest = {
            "schema_version": "1.1.0",
            "round": round_number,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "hash_algorithm": "Sort records bytewise by UTF-8 snapshot_path, then render each as '<file-sha256><two spaces><snapshot-path><LF>'; SHA-256 the concatenation; MANIFEST.json excluded",
            "snapshot_sha256": snapshot_hash,
            "files": rows,
        }
        (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        destination.parent.mkdir(parents=True)
        shutil.copytree(stage, destination)
        return snapshot_hash, len(rows)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--project", action="append", choices=PROJECTS)
    args = parser.parse_args()
    if args.round < 2:
        raise SystemExit("--round must be at least 2")
    for name in args.project or PROJECTS:
        snapshot_hash, count = freeze(name, args.round)
        print(f"{name}\t{snapshot_hash}\t{count} files")


if __name__ == "__main__":
    main()
