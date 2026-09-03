#!/usr/bin/env python3
"""Fail-closed closure audit for the ForkAudit paper package.

The audit links claims to registered evidence, resolves every registered
artifact and method source, verifies the anonymous supplement byte-for-byte,
checks its redaction boundary, and confirms the local inputs required by the
LaTeX manuscript.  A machine-readable report is written on both success and
failure; any failed invariant produces a non-zero exit status.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PAPER_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SCAN = (
    ("personal_home_path", re.compile(r"/Users/[^/\s\"']+")),
    (
        "private_remote_user_path",
        re.compile(r"/mnt/tidal-alsh-hilab/dataset/diandian/user/[^/\s\"']+"),
    ),
    ("private_job_service_hostname", re.compile(r"qs2\.devops", re.IGNORECASE)),
    ("personal_username", re.compile(r"liuhanzuo", re.IGNORECASE)),
    ("organization_name", re.compile(r"xiaohongshu", re.IGNORECASE)),
)

SENSITIVE_METADATA_KEYS = {
    "job_id",
    "trial_id",
    "queue_id",
    "cluster_id",
    "package_id",
    "pod_name",
    "node_name",
    "namespace",
    "run_url",
    "job_url",
    "trial_url",
    "web_ui_url",
}
REDACTED_METADATA_VALUE = "<REDACTED>"

BRACE_PATTERN = re.compile(r"\{([^{}]+)\}")
NUMERIC_RANGE_PATTERN = re.compile(r"(-?\d+)\.\.(-?\d+)(?:\.\.(-?\d+))?\Z")
MANIFEST_LINE_PATTERN = re.compile(r"([0-9a-f]{64})  (.+)\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

# The registry historically used ``artifacts`` as a pure local-path mapping.
# RR2 extends it with explicitly named external provenance fields.  Keep the
# split closed-world so a typo cannot silently turn a local artifact into
# unverified metadata.
LOCAL_ARTIFACT_KEYS = {
    "anonymous_round_04_replay_package",
    "anonymous_reviewer_replay",
    "anonymous_ledger",
    "detached_receipt_manifest",
    "glob",
    "integration",
    "ledger",
    "log",
    "model_load_closure",
    "original_digest_binding",
    "preregistration",
    "prior_fp32_context_margin",
    "raw_artifact_ledger",
    "root",
    "runner",
    "scope_amendment",
    "source",
    "summary",
    "validation_report",
}
POSITIVE_INTEGER_METADATA_KEYS = {
    "anonymous_round_04_manifest_bytes",
    "anonymous_round_04_manifest_files",
    "anonymous_reviewer_replay_files",
    "qs_job_id",
    "qs_trial_id",
}
EXTERNAL_STRING_METADATA_KEYS = {"qs_web_ui", "remote_run_root", "result_file"}
STRUCTURED_METADATA_KEYS = {
    "superseded_non_scientific_attempts",
    "superseded_preflight_attempts",
    "terminal_closure_rerun",
}


def sensitive_metadata_violations(value: Any, pointer: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{pointer}.{key}"
            if key.lower() in SENSITIVE_METADATA_KEYS and item != REDACTED_METADATA_VALUE:
                violations.append(child)
            violations.extend(sensitive_metadata_violations(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(sensitive_metadata_violations(item, f"{pointer}[{index}]"))
    return violations


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def expand_braces(pattern: str) -> list[str]:
    """Expand comma braces and inclusive integer ranges, recursively."""

    match = BRACE_PATTERN.search(pattern)
    if match is None:
        return [pattern]
    body = match.group(1)
    numeric = NUMERIC_RANGE_PATTERN.fullmatch(body)
    if numeric:
        start = int(numeric.group(1))
        stop = int(numeric.group(2))
        explicit_step = numeric.group(3)
        step = int(explicit_step) if explicit_step is not None else (1 if stop >= start else -1)
        if step == 0 or (stop - start) * step < 0:
            raise ValueError(f"invalid brace range {{{body}}}")
        width = max(len(numeric.group(1).lstrip("-")), len(numeric.group(2).lstrip("-")))
        values = []
        terminal = stop + (1 if step > 0 else -1)
        for number in range(start, terminal, step):
            sign = "-" if number < 0 else ""
            values.append(f"{sign}{abs(number):0{width}d}")
    elif "," in body:
        values = body.split(",")
    else:
        raise ValueError(f"unsupported brace expression {{{body}}}")

    expanded: list[str] = []
    for value in values:
        replaced = pattern[: match.start()] + value + pattern[match.end() :]
        expanded.extend(expand_braces(replaced))
    return expanded


class ClosureAudit:
    def __init__(self, paper_root: Path) -> None:
        self.paper_root = paper_root.resolve()
        self.supplement_root = (self.paper_root / "supplement_anonymous").resolve()
        self.errors: list[str] = []
        self.checks: dict[str, Any] = {}
        self.evidence_ids: set[str] = set()

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def read_json(self, path: Path, label: str) -> Any | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.fail(f"{label}: cannot read valid JSON at {path}: {exc}")
            return None

    def read_tsv(self, path: Path, required: set[str], label: str) -> list[dict[str, str]]:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                fields = set(reader.fieldnames or [])
                missing = sorted(required - fields)
                if missing:
                    self.fail(f"{label}: missing TSV columns: {', '.join(missing)}")
                    return []
                return [dict(row) for row in reader]
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            self.fail(f"{label}: cannot read TSV at {path}: {exc}")
            return []

    def audit_registry(self) -> None:
        path = self.paper_root / "evidence/experiment_registry.json"
        payload = self.read_json(path, "experiment registry")
        experiments = payload.get("experiments") if isinstance(payload, dict) else None
        if not isinstance(experiments, list):
            self.fail("experiment registry: 'experiments' must be a list")
            experiments = []

        seen: dict[str, int] = {}
        artifact_references: list[tuple[str, str, str]] = []
        artifact_set_expectations: list[tuple[str, str, int]] = []
        digest_bindings: list[tuple[str, str, str, str]] = []
        metadata_fields = 0
        for index, experiment in enumerate(experiments):
            if not isinstance(experiment, dict):
                self.fail(f"experiment registry: experiments[{index}] is not an object")
                continue
            evidence_id = experiment.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                self.fail(f"experiment registry: experiments[{index}] has no non-empty evidence_id")
                continue
            evidence_id = evidence_id.strip()
            if evidence_id in seen:
                self.fail(
                    f"experiment registry: duplicate evidence_id {evidence_id!r} "
                    f"at rows {seen[evidence_id]} and {index}"
                )
            else:
                seen[evidence_id] = index
                self.evidence_ids.add(evidence_id)
            artifacts = experiment.get("artifacts")
            if isinstance(artifacts, dict) and artifacts:
                for key, value in sorted(artifacts.items()):
                    qualified = f"{evidence_id}.artifacts.{key}"
                    if key in LOCAL_ARTIFACT_KEYS:
                        if not isinstance(value, str) or not value.strip():
                            self.fail(f"experiment registry: {qualified} is not a local path string")
                            continue
                        artifact_references.append((evidence_id, key, value.strip()))
                    elif key.endswith("_sha256"):
                        metadata_fields += 1
                        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
                            self.fail(f"experiment registry: {qualified} is not a lowercase SHA-256")
                            continue
                        bound_key = key[: -len("_sha256")]
                        bound_value = artifacts.get(bound_key)
                        if bound_key in LOCAL_ARTIFACT_KEYS and isinstance(bound_value, str):
                            digest_bindings.append((evidence_id, key, bound_value, value))
                    elif key in POSITIVE_INTEGER_METADATA_KEYS:
                        metadata_fields += 1
                        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                            self.fail(f"experiment registry: {qualified} is not a positive integer")
                    elif key == "qs_web_ui":
                        metadata_fields += 1
                        if not isinstance(value, str) or not value.startswith("https://"):
                            self.fail(f"experiment registry: {qualified} is not an HTTPS URL")
                    elif key in {"remote_run_root", "remote_run_dir"}:
                        metadata_fields += 1
                        if not isinstance(value, str) or not value.startswith("/"):
                            self.fail(f"experiment registry: {qualified} is not an absolute remote provenance path")
                    elif key == "result_file":
                        metadata_fields += 1
                        if (
                            not isinstance(value, str)
                            or not value
                            or Path(value).name != value
                            or value in {".", ".."}
                        ):
                            self.fail(f"experiment registry: {qualified} is not a safe remote basename")
                    elif key in STRUCTURED_METADATA_KEYS:
                        metadata_fields += 1
                        expected = dict if key == "terminal_closure_rerun" else list
                        if not isinstance(value, expected) or not value:
                            self.fail(
                                f"experiment registry: {qualified} must be a non-empty "
                                f"{expected.__name__}"
                            )
                    else:
                        self.fail(f"experiment registry: {qualified} has an unknown field type")
            else:
                # Snapshot-local v2 registries use explicit path lists and
                # cardinality-checked glob sets rather than the historical
                # heterogeneous ``artifacts`` mapping.
                artifact_paths = experiment.get("artifact_paths")
                artifact_sets = experiment.get("artifact_sets", [])
                if not isinstance(artifact_paths, list) or not artifact_paths:
                    self.fail(f"experiment registry: {evidence_id} has no artifact mapping")
                    continue
                for path_index, value in enumerate(artifact_paths):
                    if not isinstance(value, str) or not value.strip():
                        self.fail(
                            f"experiment registry: {evidence_id}.artifact_paths[{path_index}] "
                            "is not a local path string"
                        )
                        continue
                    artifact_references.append(
                        (evidence_id, f"artifact_paths[{path_index}]", value.strip())
                    )
                if not isinstance(artifact_sets, list):
                    self.fail(f"experiment registry: {evidence_id}.artifact_sets is not a list")
                    artifact_sets = []
                for set_index, item in enumerate(artifact_sets):
                    if not isinstance(item, dict) or set(item) != {"glob", "expected_count"}:
                        self.fail(
                            f"experiment registry: {evidence_id}.artifact_sets[{set_index}] "
                            "must contain exactly glob and expected_count"
                        )
                        continue
                    raw_glob = item.get("glob")
                    expected_count = item.get("expected_count")
                    if not isinstance(raw_glob, str) or not raw_glob.strip():
                        self.fail(
                            f"experiment registry: {evidence_id}.artifact_sets[{set_index}].glob "
                            "is not a local path string"
                        )
                        continue
                    if (
                        isinstance(expected_count, bool)
                        or not isinstance(expected_count, int)
                        or expected_count <= 0
                    ):
                        self.fail(
                            f"experiment registry: {evidence_id}.artifact_sets[{set_index}] "
                            "has invalid expected_count"
                        )
                        continue
                    artifact_references.append(
                        (evidence_id, f"artifact_sets[{set_index}]", raw_glob.strip())
                    )
                    artifact_set_expectations.append(
                        (evidence_id, raw_glob.strip(), expected_count)
                    )

        resolved = self.resolve_references(artifact_references, "registry")
        for evidence_id, raw_glob, expected_count in artifact_set_expectations:
            matches: set[Path] = set()
            try:
                candidates = self.candidate_patterns(raw_glob)
            except ValueError as exc:
                self.fail(
                    f"experiment registry: {evidence_id} has invalid artifact-set glob "
                    f"{raw_glob!r}: {exc}"
                )
                continue
            for candidate in candidates:
                for match_text in glob.glob(str(candidate), recursive=True):
                    match = Path(match_text).resolve()
                    if match.is_file() and (
                        is_within(match, self.paper_root) or is_within(match, self.supplement_root)
                    ):
                        matches.add(match)
            if len(matches) != expected_count:
                self.fail(
                    f"experiment registry: {evidence_id} artifact-set {raw_glob!r} "
                    f"expected {expected_count} files, found {len(matches)}"
                )
        verified_digests = 0
        for evidence_id, key, raw_path, expected in digest_bindings:
            if any(token in raw_path for token in ("*", "?", "[", "{")):
                self.fail(
                    f"experiment registry: {evidence_id}.artifacts.{key} cannot bind "
                    f"a glob or brace path: {raw_path!r}"
                )
                continue
            try:
                candidates = self.candidate_patterns(raw_path)
            except ValueError as exc:
                self.fail(
                    f"experiment registry: {evidence_id}.artifacts.{key} has invalid "
                    f"bound path {raw_path!r}: {exc}"
                )
                continue
            matches = [candidate.resolve() for candidate in candidates if candidate.is_file()]
            matches = sorted(set(matches))
            if len(matches) != 1:
                self.fail(
                    f"experiment registry: {evidence_id}.artifacts.{key} expected exactly "
                    f"one bound file for {raw_path!r}, found {len(matches)}"
                )
                continue
            actual = sha256(matches[0])
            if actual != expected:
                self.fail(
                    f"experiment registry: {evidence_id}.artifacts.{key} mismatch for "
                    f"{raw_path!r}: expected {expected}, got {actual}"
                )
                continue
            verified_digests += 1
        self.checks["experiment_registry"] = {
            "file": path.relative_to(self.paper_root).as_posix(),
            "experiment_count": len(experiments),
            "unique_evidence_id_count": len(self.evidence_ids),
            "evidence_ids": sorted(self.evidence_ids),
            "artifact_reference_count": len(artifact_references),
            "resolved_artifact_file_count": resolved,
            "external_metadata_field_count": metadata_fields,
            "verified_local_digest_binding_count": verified_digests,
        }

    def audit_claim_map(self) -> None:
        path = self.paper_root / "evidence/claim_evidence_map.tsv"
        rows = self.read_tsv(path, {"claim_id", "evidence_ids"}, "claim evidence map")
        unknown: list[dict[str, str]] = []
        referenced: set[str] = set()
        for row_index, row in enumerate(rows, start=2):
            claim_id = (row.get("claim_id") or "").strip()
            if not claim_id:
                self.fail(f"claim evidence map: row {row_index} has an empty claim_id")
            ids = [item.strip() for item in (row.get("evidence_ids") or "").split(";") if item.strip()]
            if not ids:
                self.fail(f"claim evidence map: {claim_id or f'row {row_index}'} has no evidence_ids")
            for evidence_id in ids:
                referenced.add(evidence_id)
                if evidence_id not in self.evidence_ids:
                    unknown.append({"claim_id": claim_id, "evidence_id": evidence_id})
                    self.fail(
                        f"claim evidence map: claim {claim_id!r} references unknown "
                        f"evidence_id {evidence_id!r}"
                    )
        self.checks["claim_evidence_map"] = {
            "file": path.relative_to(self.paper_root).as_posix(),
            "claim_count": len(rows),
            "referenced_evidence_ids": sorted(referenced),
            "unknown_references": unknown,
        }

    def audit_provenance(self) -> None:
        path = self.paper_root / "evidence/method_provenance.tsv"
        rows = self.read_tsv(path, {"method_id", "source_path"}, "method provenance")
        references: list[tuple[str, str, str]] = []
        for row_index, row in enumerate(rows, start=2):
            method_id = (row.get("method_id") or "").strip()
            source_path = (row.get("source_path") or "").strip()
            if not method_id:
                self.fail(f"method provenance: row {row_index} has an empty method_id")
            if not source_path:
                self.fail(f"method provenance: {method_id or f'row {row_index}'} has no source_path")
                continue
            references.append((method_id, "source_path", source_path))
        resolved = self.resolve_references(references, "provenance")
        self.checks["method_provenance"] = {
            "file": path.relative_to(self.paper_root).as_posix(),
            "method_count": len(rows),
            "source_reference_count": len(references),
            "resolved_source_file_count": resolved,
        }

    def candidate_patterns(self, raw: str) -> list[Path]:
        path = Path(raw)
        if path.is_absolute():
            raise ValueError("absolute paths are forbidden")
        if any(part == ".." for part in path.parts):
            raise ValueError("parent traversal is forbidden")
        if path.parts and path.parts[0] == "supplement_anonymous":
            return [self.paper_root / path]
        return [self.paper_root / path, self.supplement_root / path]

    def resolve_references(self, references: Iterable[tuple[str, str, str]], label: str) -> int:
        resolved_files: set[str] = set()
        for owner, key, raw in references:
            try:
                expanded = expand_braces(raw)
                candidates = self.candidate_patterns(raw)
            except ValueError as exc:
                self.fail(f"{label}: {owner}.{key} path {raw!r} is invalid: {exc}")
                continue

            # A brace list/range denotes an explicit required set: every expansion
            # must resolve.  Ordinary shell wildcards may resolve to one or more files.
            for expansion in expanded:
                try:
                    expansion_candidates = self.candidate_patterns(expansion)
                except ValueError as exc:
                    self.fail(f"{label}: {owner}.{key} path {expansion!r} is invalid: {exc}")
                    continue
                matches: list[Path] = []
                for candidate in expansion_candidates:
                    for match_text in glob.glob(str(candidate), recursive=True):
                        match = Path(match_text).resolve()
                        if not (is_within(match, self.paper_root) or is_within(match, self.supplement_root)):
                            self.fail(f"{label}: {owner}.{key} resolves outside package roots: {match}")
                            continue
                        if match.is_file() or match.is_dir():
                            matches.append(match)
                unique_matches = sorted(set(matches))
                if not unique_matches:
                    self.fail(f"{label}: {owner}.{key} path/glob has no match: {expansion!r}")
                for match in unique_matches:
                    if match.is_dir():
                        # Directory roots are valid registry artifacts; bind every
                        # contained file so the audit still reports concrete closure.
                        files = sorted(item for item in match.rglob("*") if item.is_file())
                        if not files:
                            self.fail(f"{label}: {owner}.{key} resolves to an empty directory: {match}")
                        for item in files:
                            resolved_files.add(item.relative_to(self.paper_root).as_posix())
                    else:
                        resolved_files.add(match.relative_to(self.paper_root).as_posix())
            if not candidates:  # Defensive: candidate generation must never be empty.
                self.fail(f"{label}: {owner}.{key} produced no candidate roots")
        return len(resolved_files)

    def audit_manifest(self) -> None:
        manifest = self.supplement_root / "ANONYMOUS_MANIFEST.sha256"
        entries: dict[str, str] = {}
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            self.fail(f"anonymous manifest: cannot read {manifest}: {exc}")
            lines = []
        for line_number, line in enumerate(lines, start=1):
            match = MANIFEST_LINE_PATTERN.fullmatch(line)
            if match is None:
                self.fail(f"anonymous manifest: malformed line {line_number}")
                continue
            expected, relative = match.groups()
            if relative in entries:
                self.fail(f"anonymous manifest: duplicate path {relative!r}")
                continue
            relative_path = Path(relative)
            if relative_path.is_absolute() or any(part == ".." for part in relative_path.parts):
                self.fail(f"anonymous manifest: unsafe path {relative!r}")
                continue
            target = (self.supplement_root / relative_path).resolve()
            if not is_within(target, self.supplement_root):
                self.fail(f"anonymous manifest: path escapes supplement root: {relative!r}")
                continue
            entries[relative] = expected
            if not target.is_file():
                self.fail(f"anonymous manifest: missing file {relative!r}")
                continue
            actual = sha256(target)
            if actual != expected:
                self.fail(
                    f"anonymous manifest: SHA-256 mismatch for {relative!r}: "
                    f"expected {expected}, got {actual}"
                )

        actual_files = {
            path.relative_to(self.supplement_root).as_posix()
            for path in self.supplement_root.rglob("*")
            if path.is_file() and path != manifest
        } if self.supplement_root.is_dir() else set()
        unmanifested = sorted(actual_files - set(entries))
        missing_from_disk = sorted(set(entries) - actual_files)
        if unmanifested:
            self.fail(f"anonymous manifest: unmanifested files: {', '.join(unmanifested)}")
        if missing_from_disk:
            self.fail(f"anonymous manifest: entries missing from disk: {', '.join(missing_from_disk)}")
        self.checks["anonymous_manifest"] = {
            "file": manifest.relative_to(self.paper_root).as_posix(),
            "manifest_sha256": sha256(manifest) if manifest.is_file() else None,
            "entry_count": len(entries),
            "actual_file_count_excluding_manifest": len(actual_files),
            "unmanifested_files": unmanifested,
            "missing_files": missing_from_disk,
        }

    def audit_anonymity(self) -> None:
        text_files = 0
        bytes_scanned = 0
        violations: list[dict[str, Any]] = []
        if not self.supplement_root.is_dir():
            self.fail(f"anonymity scan: missing supplement directory {self.supplement_root}")
        else:
            for path in sorted(item for item in self.supplement_root.rglob("*") if item.is_file()):
                try:
                    content = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                except OSError as exc:
                    self.fail(f"anonymity scan: cannot read {path}: {exc}")
                    continue
                text_files += 1
                bytes_scanned += path.stat().st_size
                for rule, pattern in FORBIDDEN_SCAN:
                    match = pattern.search(content)
                    if match:
                        line = content.count("\n", 0, match.start()) + 1
                        relative = path.relative_to(self.supplement_root).as_posix()
                        violations.append({"file": relative, "line": line, "rule": rule})
                        self.fail(f"anonymity scan: {rule} matched {relative}:{line}")
                if path.suffix.lower() == ".json":
                    try:
                        value = json.loads(content)
                    except json.JSONDecodeError as exc:
                        relative = path.relative_to(self.supplement_root).as_posix()
                        violations.append({
                            "file": relative,
                            "line": exc.lineno,
                            "rule": "invalid_json",
                        })
                        self.fail(f"anonymity scan: invalid JSON in {relative}:{exc.lineno}")
                    else:
                        relative = path.relative_to(self.supplement_root).as_posix()
                        for pointer in sensitive_metadata_violations(value):
                            violations.append({
                                "file": relative,
                                "json_pointer": pointer,
                                "rule": "unredacted_sensitive_metadata",
                            })
                            self.fail(
                                "anonymity scan: sensitive metadata is not redacted "
                                f"in {relative}:{pointer}"
                            )
        self.checks["anonymity_scan"] = {
            "root": self.supplement_root.relative_to(self.paper_root).as_posix(),
            "text_files_checked": text_files,
            "bytes_scanned": bytes_scanned,
            "forbidden_rules": [name for name, _ in FORBIDDEN_SCAN],
            "sensitive_metadata_keys": sorted(SENSITIVE_METADATA_KEYS),
            "required_sensitive_metadata_value": REDACTED_METADATA_VALUE,
            "violations": violations,
        }

    def audit_latex_inputs(self) -> None:
        main = self.paper_root / "main.tex"
        try:
            source = main.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self.fail(f"LaTeX inputs: cannot read {main}: {exc}")
            source = ""

        required: set[str] = set()
        for value in re.findall(r"\\(?:input|IfFileExists)\s*\{([^}]+)\}", source):
            if value.endswith((".tex", ".sty", ".bst", ".bib", ".pdf", ".png", ".jpg", ".jpeg")):
                required.add(value)
            else:
                required.add(value + ".tex")
        for value in re.findall(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}", source):
            candidate = Path(value)
            if candidate.suffix:
                required.add(value)
            else:
                found = False
                for extension in (".pdf", ".png", ".jpg", ".jpeg"):
                    if (self.paper_root / f"{value}{extension}").is_file():
                        required.add(f"{value}{extension}")
                        found = True
                        break
                if not found:
                    required.add(value + ".pdf")
        for group in re.findall(r"\\bibliography\s*\{([^}]+)\}", source):
            required.update(f"{name.strip()}.bib" for name in group.split(",") if name.strip())
        for value in re.findall(r"\\bibliographystyle\s*\{([^}]+)\}", source):
            required.add(value + ".bst")
        for group in re.findall(r"\\usepackage(?:\[[^]]*\])?\s*\{([^}]+)\}", source):
            for package in (item.strip() for item in group.split(",")):
                if package.startswith("iclr") or (self.paper_root / f"{package}.sty").is_file():
                    required.add(package + ".sty")

        # These are the four local input classes that the submission must bind.
        classes = {
            "style": sorted(path for path in required if path.endswith((".sty", ".bst"))),
            "bibliography": sorted(path for path in required if path.endswith(".bib")),
            "figures": sorted(path for path in required if path.startswith("figures/")),
            "tables": sorted(path for path in required if path.startswith("tables/")),
            "other_inputs": sorted(
                path for path in required
                if not path.endswith((".sty", ".bst", ".bib"))
                and not path.startswith(("figures/", "tables/"))
            ),
        }
        for name in ("style", "bibliography", "figures", "tables"):
            if not classes[name]:
                self.fail(f"LaTeX inputs: no local {name} dependency was discovered in main.tex")

        missing = []
        for relative in sorted(required):
            target = (self.paper_root / relative).resolve()
            if not is_within(target, self.paper_root) or not target.is_file():
                missing.append(relative)
                self.fail(f"LaTeX inputs: missing required local file {relative!r}")
        self.checks["latex_inputs"] = {
            "file": main.relative_to(self.paper_root).as_posix(),
            "required_by_class": classes,
            "required_file_count": len(required),
            "missing": missing,
        }

    def run(self) -> dict[str, Any]:
        self.audit_registry()
        self.audit_claim_map()
        self.audit_provenance()
        self.audit_manifest()
        self.audit_anonymity()
        self.audit_latex_inputs()
        return {
            "schema_version": "1.0.0",
            "status": "passed" if not self.errors else "failed",
            "paper_root": ".",
            "checks": self.checks,
            "error_count": len(self.errors),
            "errors": self.errors,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paper_root = args.paper_root.resolve()
    output = args.output.resolve() if args.output else paper_root / "generated/package_closure_audit.json"
    if not is_within(output, paper_root):
        print("error: audit output must be inside the paper root", file=sys.stderr)
        raise SystemExit(2)

    audit = ClosureAudit(paper_root).run()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "error_count": audit["error_count"],
        "output": output.as_posix(),
        "output_sha256": sha256(output),
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if audit["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
