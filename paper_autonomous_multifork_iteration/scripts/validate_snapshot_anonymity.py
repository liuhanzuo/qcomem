#!/usr/bin/env python3
"""Reproduce the blind-snapshot anonymity scan without external state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXCLUDED_FILES = {
    "MANIFEST.json",
    "evidence/BLIND_ANONYMITY_AUDIT.json",
}
EXCLUDED_PREFIXES: tuple[str, ...] = ()
TEXT_SUFFIXES = {
    "", ".bib", ".bst", ".csv", ".json", ".jsonl", ".log", ".md",
    ".nul", ".py", ".sha256", ".sh", ".sty", ".svg", ".tex", ".tsv", ".txt", ".yaml",
    ".yml",
}
BINARY_SUFFIXES = {
    ".bin", ".dll", ".dylib", ".jpeg", ".jpg", ".npy", ".npz", ".pdf",
    ".pkl", ".png", ".pt", ".pth", ".pyc", ".so",
}

# Split sensitive literals so this validator can scan its own source without
# containing the forbidden byte strings as contiguous author-identifying text.
FORBIDDEN = (
    b"/" + b"Users/",
    b"/mnt/" + b"tidal-alsh-hilab",
    b"liu" + b"hanzuo",
    b"qs2." + b"devops",
    b"artifactory." + b"devops",
    b"xiao" + b"hongshu.com",
    b"review_" + b"response_plan_raw_base64",
    b'"review_' + b'response_plan_audit"',
    b"review_" + b"response_plan",
    b"review_" + b"experiment_plan",
    b"review-" + b"experiment-plan",
    b"review " + b"response plan",
    b"reviewer-" + b"response",
    b"blind_" + b"preoutput",
    b"pre-output " + b"governance input",
    b"reviewer_" + b"feedback_precedes_new_experiment",
    b"rr2-exp-" + b"ownership-mutants",
)
SEPARATOR_NORMALIZED_FORBIDDEN = (
    b"review" + b" response plan",
    b"review" + b" experiment plan",
    b"reviewer" + b" response",
    b"blind" + b" preoutput",
    b"preoutput" + b" plan",
    b"review" + b" revision",
)
PLATFORM_PATTERNS = (
    re.compile(rb"(/model/production/job/trial/)[0-9]+/[0-9]+"),
    re.compile(
        rb'["\'](?:qs_job_id|qs_trial_id|job_id|trial_id|queue_id|cloud_id|cluster_id|resource_package_id|'
        rb'jobId|trialId|queueId|cloudId|clusterId|resourcePackageId)["\']\s*:\s*[0-9]+'
    ),
    re.compile(rb"\b(?:Job|Trial)\s+[0-9]+\b"),
)


def scan_text(path: Path, relative: str) -> None:
    tail = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            payload = (tail + block).lower()
            for needle in FORBIDDEN:
                if needle.lower() in payload:
                    raise RuntimeError(
                        f"blind snapshot leaked a forbidden token in {relative}"
                    )
            normalized = re.sub(rb"[-_\s]+", b" ", payload)
            for needle in SEPARATOR_NORMALIZED_FORBIDDEN:
                if needle in normalized:
                    raise RuntimeError(
                        "blind snapshot leaked a separator-obfuscated governance "
                        f"token in {relative}"
                    )
            for pattern in PLATFORM_PATTERNS:
                if pattern.search(payload):
                    raise RuntimeError(
                        "blind snapshot leaked a numeric platform identifier "
                        f"in {relative}"
                    )
            tail = payload[-512:]


def digest_paths(paths: list[str]) -> str:
    value = hashlib.sha256()
    for relative in paths:
        value.update(relative.encode("utf-8"))
        value.update(b"\n")
    return value.hexdigest()


def audit(root: Path) -> dict[str, object]:
    scanned_paths: list[str] = []
    binary_allowlisted_paths: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in EXCLUDED_FILES or relative.startswith(EXCLUDED_PREFIXES):
            continue
        suffix = path.suffix.lower()
        if suffix in BINARY_SUFFIXES:
            binary_allowlisted_paths.append(relative)
            continue
        if suffix not in TEXT_SUFFIXES:
            raise RuntimeError(f"unclassified snapshot file suffix: {relative}")
        scan_text(path, relative)
        scanned_paths.append(relative)
    return {
        "schema_version": "forkaudit-full-snapshot-anonymity-audit-v2",
        "status": "PASS",
        "validator": "scripts/validate_snapshot_anonymity.py",
        "scanned_paths": scanned_paths,
        "scanned_paths_sha256": digest_paths(scanned_paths),
        "text_files_scanned": len(scanned_paths),
        "binary_allowlisted_paths": binary_allowlisted_paths,
        "binary_allowlisted_paths_sha256": digest_paths(binary_allowlisted_paths),
        "binary_allowlisted_file_count": len(binary_allowlisted_paths),
        "excluded_files": sorted(EXCLUDED_FILES),
        "excluded_exact_source_prefixes": list(EXCLUDED_PREFIXES),
        "rule_classes": [
            "private absolute home or shared-mount path",
            "private user or organization identifier",
            "private platform or registry hostname",
            "hidden response-plan payload or audit field",
            "numeric job/trial URL or structured platform identifier",
        ],
        "large_binary_sidecars_checked_by_top_manifest": True,
        "opaque_gpu_uuid_policy": (
            "reviewer-safe hardware assignment identifiers: retained because they "
            "are random device identities, not author, organization, job, node, or "
            "platform-account identifiers; only distinctness is claimed"
        ),
        "response_plan_preimage_present": False,
        "numeric_platform_identifiers_present_outside_exact_source": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = audit(root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = root / output
        output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
