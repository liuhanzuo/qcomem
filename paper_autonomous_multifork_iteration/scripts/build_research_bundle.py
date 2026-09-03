#!/usr/bin/env python3
"""Build deterministic ForkAudit paper-material and anonymous-supplement ZIPs.

The private research bundle includes the paper sources, review audit trail,
official ICLR template, anonymous derivative, and the original raw output
artifacts used by the manuscript.  It intentionally does not redistribute
model weights or PG-19 text; the run ledgers bind those external inputs.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable


PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
FULL_ZIP = REPO / "ForkAudit_ICLR2026_paper_materials.zip"
ANON_ZIP = REPO / "ForkAudit_ICLR2026_anonymous_supplement.zip"

PAPER_ROOT_FILES = {
    "main.tex",
    "main.pdf",
    "main.bbl",
    "references.bib",
    "iclr2026_conference.sty",
    "iclr2026_conference.bst",
    "math_commands.tex",
    "fancyhdr.sty",
    "natbib.sty",
}
PAPER_DIRS = {
    "build",
    "evidence",
    "external_templates",
    "figures",
    "generated",
    "literature",
    "scripts",
    "skill_release",
    "state",
    "supplement_anonymous",
    "tables",
}
RAW_RUNS = (
    "gpu-qwen35-vllm-paged-multifork-resident-20260814a",
    "gpu-qwen35-vllm-paged-fair-v2-20260814c",
    "gpu-qwen35-vllm-paged-q16-formal-negative-20260814b",
)
REPORTS = (
    "RESULTS_GPU_QWEN35_VLLM_PAGED_MULTIFORK_RESIDENT_2026-08-14_ZH.md",
    "RESULTS_GPU_QWEN35_VLLM_PAGED_FAIR_V2_2026-08-14_ZH.md",
    "RESULTS_GPU_QWEN35_VLLM_PAGED_Q16_FORMAL_NEGATIVE_2026-08-14_ZH.md",
)
CONFIGS = (
    "qs/qcomem-qwen35-vllm-paged-multifork-resident-20260814a.yaml",
    "qs/qcomem-qwen35-vllm-paged-fair-v2-20260814c.yaml",
    "qs/qcomem-qwen35-vllm-paged-q16-formal-20260814b.yaml",
    "gpu/build_qcomem_qwen35_vllm_paged_multifork_manifest.py",
    "gpu/qcomem_qwen35_vllm_paged_multifork_resident_protocol_manifest.json",
)
SKILL_FILES = (
    PAPER / "skill_release/autonomous-paper-agent-v2/SKILL.md",
    PAPER / "skill_release/autonomous-paper-agent-v2/references/iclr-kv-cache-calibration.md",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def files_under(root: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def paper_entries() -> dict[str, Path]:
    entries: dict[str, Path] = {}
    prefix = "ForkAudit_ICLR2026/paper"
    for name in sorted(PAPER_ROOT_FILES):
        source = PAPER / name
        if not source.is_file():
            raise RuntimeError(f"missing paper file: {source}")
        entries[f"{prefix}/{name}"] = source
    for directory in sorted(PAPER_DIRS):
        root = PAPER / directory
        if not root.is_dir():
            raise RuntimeError(f"missing paper directory: {root}")
        for source in files_under(root):
            relative = source.relative_to(PAPER).as_posix()
            entries[f"{prefix}/{relative}"] = source

    # Keep review decisions and immutable snapshot manifests, not duplicated
    # 200+ MiB submission payloads that are already represented by the paper
    # sources and supplement above.
    review_root = PAPER / "review"
    if review_root.is_dir():
        for source in files_under(review_root):
            relative = source.relative_to(PAPER).as_posix()
            parts = source.relative_to(review_root).parts
            inside_submission = "submission" in parts
            if inside_submission and source.name != "MANIFEST.json":
                continue
            entries[f"{prefix}/{relative}"] = source
    return entries


def full_entries() -> dict[str, Path]:
    entries = paper_entries()
    prefix = "ForkAudit_ICLR2026"
    for run in RAW_RUNS:
        root = REPO / "results" / run
        if not root.is_dir():
            raise RuntimeError(f"missing original run artifacts: {root}")
        for source in files_under(root):
            relative = source.relative_to(root).as_posix()
            entries[f"{prefix}/original_raw_runs/{run}/{relative}"] = source
    for name in REPORTS:
        source = REPO / name
        if not source.is_file():
            raise RuntimeError(f"missing report: {source}")
        entries[f"{prefix}/reports/{name}"] = source
    for relative in CONFIGS:
        source = REPO / relative
        if not source.is_file():
            raise RuntimeError(f"missing execution configuration: {source}")
        entries[f"{prefix}/execution_configs/{relative}"] = source
    for source in SKILL_FILES:
        if not source.is_file():
            raise RuntimeError(f"missing skill snapshot: {source}")
        entries[f"{prefix}/review_rubric/{source.name}"] = source
    return entries


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def write_zip(output: Path, entries: dict[str, Path], readme: str) -> dict[str, object]:
    payloads: dict[str, bytes] = {
        name: source.read_bytes() for name, source in sorted(entries.items())
    }
    root = next(iter(sorted(payloads))).split("/", 1)[0]
    payloads[f"{root}/README_BUNDLE.md"] = readme.encode("utf-8")
    manifest_name = f"{root}/MANIFEST.sha256"
    manifest = "".join(
        f"{sha256_bytes(value)}  {name}\n"
        for name, value in sorted(payloads.items())
    ).encode("utf-8")
    payloads[manifest_name] = manifest

    with zipfile.ZipFile(output, "w", allowZip64=True, compresslevel=9) as archive:
        for name, value in sorted(payloads.items()):
            archive.writestr(zip_info(name), value)

    with zipfile.ZipFile(output, "r") as archive:
        names = set(archive.namelist())
        if names != set(payloads):
            raise RuntimeError("ZIP entry set differs from the requested bundle")
        for name, expected in payloads.items():
            actual = archive.read(name)
            if sha256_bytes(actual) != sha256_bytes(expected):
                raise RuntimeError(f"ZIP replay mismatch: {name}")
    return {
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
        "entries": len(payloads),
        "manifest_sha256": sha256_bytes(manifest),
    }


def main() -> None:
    full_readme = """# ForkAudit ICLR 2026 research bundle

This is the complete, non-anonymous research bundle. It contains the English
ICLR manuscript and build inputs, figures/tables and derivation scripts,
anonymous supplement, independent review audit trail, official template ZIP,
the three original run-artifact directories used by the paper, their Chinese
reports, launch configurations, frozen protocol manifest, and a snapshot of
the ICLR-aligned autonomous-paper review skill, including its policy for using
ImageGen on non-evidentiary teasers and conceptual architecture illustrations.

The original experimental *outputs* are included byte-for-byte. Model weights
and PG-19 text are not redistributed; their expected digests and access
boundaries remain in the run ledgers. This archive is not double-blind because
the original run artifacts may contain local or infrastructure metadata.
"""
    anonymous_readme = """# ForkAudit ICLR 2026 anonymous supplement ZIP

This ZIP is a deterministic packaging of `supplement_anonymous/`. Its internal
`ANONYMOUS_MANIFEST.sha256` binds every included derivative file except the
manifest itself. Infrastructure identifiers and private paths are redacted.
The supplement contains model-output/state digests and runtime equality flags,
not full tensors, model weights, or PG-19 text.
"""
    full = write_zip(FULL_ZIP, full_entries(), full_readme)
    anonymous_entries = {
        f"ForkAudit_ICLR2026_anonymous_supplement/{source.relative_to(PAPER / 'supplement_anonymous').as_posix()}": source
        for source in files_under(PAPER / "supplement_anonymous")
    }
    anonymous = write_zip(ANON_ZIP, anonymous_entries, anonymous_readme)
    print(json.dumps({"full_bundle": full, "anonymous_supplement": anonymous}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
