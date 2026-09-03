#!/usr/bin/env python3
"""Build the self-contained blind Round-5 reviewer snapshot."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clone(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        subprocess.run(["cp", "-cR", str(source), str(destination)], check=True)
    else:
        subprocess.run(["cp", "-c", str(source), str(destination)], check=True)


def write_complete_package_manifest(
    root: Path, *, schema_version: str, parent_manifest_sha256: str
) -> str:
    excluded = {"MANIFEST.json", "MANIFEST.sha256"}
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": schema_version,
        "parent_manifest_sha256": parent_manifest_sha256,
        "excludes": sorted(excluded),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    manifest_path = root / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha256(manifest_path)
    (root / "MANIFEST.sha256").write_text(
        f"{digest}  MANIFEST.json\n", encoding="ascii"
    )
    return digest


def refresh_plain_package_manifest(root: Path) -> str:
    """Refresh a manifest-first package without changing its schema."""

    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded = set(manifest.get("excludes", ["MANIFEST.json", "MANIFEST.sha256"]))
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest["files"] = rows
    manifest["file_count"] = len(rows)
    manifest["total_bytes"] = sum(row["bytes"] for row in rows)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha256(manifest_path)
    (root / "MANIFEST.sha256").write_text(
        f"{digest}  MANIFEST.json\n", encoding="ascii"
    )
    return digest


def refresh_rr2_snapshot_metadata(staging: Path) -> None:
    """Bind reviewer summaries to the derivative actually shipped."""

    rr2_root = staging / "evidence/round_04_rr2_package"
    manifest_path = rr2_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = sha256(manifest_path)
    integrated_path = staging / "evidence/integrated_results.json"
    integrated = json.loads(integrated_path.read_text(encoding="utf-8"))
    primary = integrated["primary_factorial_and_replay"]
    if "package_manifest_sha256" in primary:
        primary["source_complete_package_manifest_sha256"] = primary.pop(
            "package_manifest_sha256"
        )
        primary["source_complete_package_files"] = primary.pop("package_files")
        primary["source_complete_package_bytes"] = primary.pop("package_bytes")
    required_source_complete = {
        "source_complete_package_manifest_sha256",
        "source_complete_package_files",
        "source_complete_package_bytes",
    }
    if not required_source_complete.issubset(primary):
        raise RuntimeError("RR2 source-complete metadata is incomplete")
    primary["reviewer_derivative_manifest_sha256"] = manifest_sha
    primary["reviewer_derivative_parent_manifest_sha256"] = manifest[
        "parent_manifest_sha256"
    ]
    primary["reviewer_derivative_files"] = manifest["file_count"]
    primary["reviewer_derivative_bytes"] = manifest["total_bytes"]
    primary.pop("package_parent_manifest_sha256", None)
    primary.pop("blind_derivative_files", None)
    integrated_path.write_text(
        json.dumps(integrated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    scope = (
        "# Round-4 manifest scopes\n\n"
        "The author-side complete package has 628 entries and 892,144,066 bytes; "
        "its manifest SHA-256 is "
        f"`{primary['source_complete_package_manifest_sha256']}`.  The blind "
        "reviewer derivative omits the author-side governance preimage and regenerates "
        "governance receipts without changing scientific raw artifacts.  It has "
        f"{manifest['file_count']} entries and {manifest['total_bytes']:,} bytes; "
        f"its manifest SHA-256 is `{manifest_sha}` and its parent is "
        f"`{manifest['parent_manifest_sha256']}`.  The replay verifies this "
        "reviewer-derivative manifest before reconstructing the registered results.\n"
    )
    (staging / "evidence/round_04_manifest_scope.md").write_text(
        scope, encoding="utf-8"
    )


def prepare_rr2_blind_derivative(root: Path) -> None:
    """Omit the hidden response-plan preimage while retaining run authority."""

    response_plan = (
        root
        / "executed_inputs/qcomem_gpu_forkaudit_rr2_formal_20260818w-inputs"
        / "review-experiment-response-plan.json"
    )
    expected = "e2be05e198d6c86276f229d4e862c3579c65e311d07c219e9e9c792a390cbfbb"
    if not response_plan.is_file() or sha256(response_plan) != expected:
        raise RuntimeError("RR2 response-plan preimage authority drift")
    response_plan.unlink()
    original_digest = expected
    hidden_terms = (
        "review_response_plan",
        "review_experiment_plan",
        "review-response experiment plan",
        "reviewer-response experiment plan",
        "review experiment plan",
        "response_plan",
        "review_response",
        "review-experiment-plan",
        "response-plan",
        "blind_preoutput",
        "blind-preoutput",
        "preoutput_plan",
        "preoutput-plan",
        "review_revision",
        "review-revision",
        "review revision",
    )

    def is_hidden_text(value: str) -> bool:
        lowered = value.lower()
        return any(term in lowered for term in hidden_terms)

    def neutralize_text(value: str) -> str:
        value = re.sub(
            r"review[-_ ]*experiment[-_ ]*plan",
            "frozen_input_contract",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"(?:review(?:er)?[-_ ]*response|response)[-_ ]*plan",
            "frozen_input_contract",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"(?:blind[-_ ]*)?preoutput[-_ ]*plan",
            "frozen_input_contract",
            value,
            flags=re.IGNORECASE,
        )
        replacements = (
            (original_digest, "<REDACTED_HIDDEN_INPUT_SHA256>"),
            ("reviewer-response experiment plan", "frozen protocol input"),
            ("review-response experiment plan", "frozen protocol input"),
            ("review response plan", "frozen protocol input"),
            ("review-experiment-plan", "frozen-input-contract"),
            ("review_experiment_plan", "frozen_input_contract"),
            ("review_response_plan", "frozen_input_contract"),
            ("blind-preoutput-plan", "frozen-input-contract"),
            ("blind_preoutput_plan", "frozen_input_contract"),
            ("preoutput-plan", "frozen-input-contract"),
            ("preoutput_plan", "frozen_input_contract"),
            ("response-plan", "frozen-input-contract"),
            ("response_plan", "frozen_input_contract"),
            ("review-response", "frozen-protocol"),
            ("reviewer-response", "frozen-protocol"),
            ("reviewer response", "frozen protocol"),
            ("review_response", "frozen_protocol"),
            ("/users/", "<absolute-home>/"),
            ("/mnt/tidal-alsh-hilab", "<private-mount>"),
            ("liuhanzuo", "<anonymous-user>"),
            ("qs2.devops.xiaohongshu.com", "<private-platform>"),
            ("artifactory.devops.xiaohongshu.com", "<private-registry>"),
            ("qs2.devops", "<private-platform>"),
            ("artifactory.devops", "<private-registry>"),
            ("xiaohongshu.com", "<private-organization>"),
            ("reviewer-triggered", "auditable"),
            ("review-triggered", "auditable"),
            ("review revision", "audit protocol"),
            ("review-revision", "audit-protocol"),
            ("review_revision", "audit_protocol"),
        )
        for old, new in replacements:
            value = re.sub(re.escape(old), new, value, flags=re.IGNORECASE)
        return value

    def without_hidden_planning(value):
        if isinstance(value, dict):
            return {
                key: without_hidden_planning(item)
                for key, item in value.items()
                if not is_hidden_text(key)
            }
        if isinstance(value, list):
            return [
                without_hidden_planning(item)
                for item in value
                if not (isinstance(item, str) and is_hidden_text(item))
            ]
        if isinstance(value, str):
            return neutralize_text(value)
        return value

    transformation_rows: list[dict[str, object]] = []
    regenerated_manifest_paths = {
        root / "MANIFEST.json",
        root / "derived/release_manifest.json",
    }

    def scientific_numeric_leaf_digest(
        raw: bytes, ignored_numeric_keys: frozenset[str] = frozenset()
    ) -> str:
        value = without_hidden_planning(json.loads(raw))
        rows: list[tuple[str, int | float]] = []

        def walk(node, path: str) -> None:
            if isinstance(node, dict):
                for key in sorted(node):
                    if key in {
                        "source_closure_file_count",
                        "execution_source_closure_file_count",
                        "reviewer_derivative_source_closure_file_count",
                    } or key in ignored_numeric_keys:
                        continue
                    walk(node[key], f"{path}/{key}")
            elif isinstance(node, list):
                for index, item in enumerate(node):
                    walk(item, f"{path}/{index}")
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                rows.append((path, node))

        walk(value, "")
        encoded = json.dumps(
            rows, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def record_change(path: Path, original: bytes) -> None:
        current = path.read_bytes()
        if current == original:
            return
        row: dict[str, object] = {
            "path": path.relative_to(root).as_posix(),
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "reviewer_derivative_sha256": hashlib.sha256(current).hexdigest(),
            "scientific_numeric_payload_modified": False,
            "transformation": "remove author-side governance metadata only",
        }
        if path.suffix.lower() == ".json":
            ignored_numeric_keys = (
                frozenset({"bytes"})
                if path.name == "detached-receipt-manifest.json"
                else frozenset()
            )
            original_numeric = scientific_numeric_leaf_digest(
                original, ignored_numeric_keys
            )
            derivative_numeric = scientific_numeric_leaf_digest(
                current, ignored_numeric_keys
            )
            if original_numeric != derivative_numeric:
                raise RuntimeError(
                    f"RR2 scientific numeric leaves changed during redaction: {path}"
                )
            row["scientific_numeric_leaf_sha256"] = derivative_numeric
        transformation_rows.append(row)

    changed_raw_paths: set[Path] = set()
    for path in sorted(root.rglob("*.json")):
        original = path.read_bytes()
        if not any(term.encode("utf-8") in original.lower() for term in hidden_terms):
            continue
        value = without_hidden_planning(json.loads(original))
        if (
            path.parent.name == "shards"
            and path.name.startswith("forkaudit-shard-")
        ):
            protocol_sha = hashlib.sha256(
                json.dumps(
                    value["protocol_config"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            value["protocol_config_sha256"] = protocol_sha
            value["frozen_identity"]["protocol_config_sha256"] = protocol_sha
            encoded = (
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            ).encode("utf-8")
            changed_raw_paths.add(path)
        else:
            encoded = (
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
        path.write_bytes(encoded)
        if path not in regenerated_manifest_paths:
            record_change(path, original)

    source_root = root / "executed_source/gpu"
    omitted_governance_sources = {
        "FORKAUDIT_REVIEW_REVISION_PROTOCOL_ZH.md",
        "build_qcomem_qwen35_forkaudit_review_manifest.py",
        "launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh",
        "test_launch_qcomem_qwen35_forkaudit_review_revision.py",
        "test_run_qcomem_qwen35_forkaudit_review_revision.py",
    }
    for name in omitted_governance_sources:
        path = source_root / name
        if not path.is_file():
            raise RuntimeError(
                f"RR2 governance source missing before blind omission: {name}"
            )
        path.unlink()

    planning_markers = (
        "blind_preoutput",
        "preoutput_plan",
        "source_round",
        "rr2-exp-ownership-mutants",
        "experiment_required",
        "reviewer_feedback_precedes_new_experiment",
    )

    def strip_planning_nodes(source: str, *, label: str) -> str:
        tree = ast.parse(source, filename=label)
        lines = source.splitlines(keepends=True)
        omitted: list[tuple[int, int]] = []
        for node in tree.body:
            start = getattr(node, "lineno", 1) - 1
            end = getattr(node, "end_lineno", start + 1)
            segment = "".join(lines[start:end]).lower()
            if any(marker in segment for marker in planning_markers):
                omitted.append((start, end))
            elif isinstance(node, ast.If) and "__name__" in segment:
                omitted.append((start, end))
        for start, end in reversed(omitted):
            lines[start:end] = ["\n"]
        derivative = "".join(lines)
        compile(derivative, label, "exec")
        lowered = derivative.lower()
        if any(marker in lowered for marker in planning_markers):
            raise RuntimeError(
                f"RR2 planning node remained after blind stripping: {label}"
            )
        return derivative

    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".py", ".sh"}:
            continue
        original = path.read_bytes()
        text = original.decode("utf-8")
        derivative = neutralize_text(text)
        derivative = derivative.replace(
            "round-2 pre-output governance input", "pre-output governance input"
        )
        if path.name == "run_qcomem_qwen35_forkaudit_review_revision.py":
            derivative = strip_planning_nodes(
                derivative, label=path.relative_to(root).as_posix()
            )
        if derivative != text:
            path.write_text(derivative, encoding="utf-8")
            record_change(path, original)

    source_renames = {
        "run_qcomem_qwen35_forkaudit_review_revision.py":
            "run_qcomem_qwen35_forkaudit_audit_protocol.py",
    }
    for old_name, new_name in source_renames.items():
        old_path = source_root / old_name
        new_path = source_root / new_name
        if not old_path.is_file() or new_path.exists():
            raise RuntimeError(f"RR2 reviewer-source rename drift: {old_name}")
        old_path.rename(new_path)
        old_relative = f"executed_source/gpu/{old_name}"
        new_relative = f"executed_source/gpu/{new_name}"
        for row in transformation_rows:
            if row["path"] == old_relative:
                row["path"] = new_relative

    # Rebind the reviewer-source derivative while retaining every method body
    # used by the raw-first numerical replay.
    code_ledger = root / "upstream/preregistration/code.sha256"
    original_code_ledger = code_ledger.read_bytes()
    code_rows = []
    for line in original_code_ledger.decode("utf-8").splitlines():
        _, relative = line.split(None, 1)
        relative = relative.strip()
        if Path(relative).name in omitted_governance_sources:
            continue
        renamed = source_renames.get(Path(relative).name)
        if renamed is not None:
            relative = f"./{renamed}"
        code_rows.append(f"{sha256(source_root / relative)}  {relative}")
    code_ledger.write_text("\n".join(code_rows) + "\n", encoding="utf-8")
    record_change(code_ledger, original_code_ledger)
    derivative_code_ledger_sha = sha256(code_ledger)
    code_files_nul = root / "upstream/preregistration/code-files.nul"
    code_files_nul.write_bytes(
        b"".join(
            relative.encode("utf-8") + b"\0"
            for _, relative in (row.split(None, 1) for row in code_rows)
        )
    )

    # Do not ship the original 34-entry executed-input ledger: its omitted
    # filenames reconstruct the author-side governance namespace.  The
    # reviewer derivative is already bound by the 29-entry ledger above.
    executed_input_ledger = (
        root
        / "executed_inputs/qcomem_gpu_forkaudit_rr2_formal_20260818w-inputs"
        / "code.sha256"
    )
    if not executed_input_ledger.is_file():
        raise RuntimeError("RR2 original executed-input code ledger is missing")
    executed_input_ledger.unlink()

    registry_entry_path = root / "experiment_registry_entry.json"
    registry_entry_original = registry_entry_path.read_bytes()
    registry_entry = json.loads(registry_entry_original)
    registry_entry["execution_source_closure_file_count"] = registry_entry.pop(
        "source_closure_file_count"
    )
    registry_entry["reviewer_derivative_source_closure_file_count"] = len(code_rows)
    registry_entry_path.write_text(
        json.dumps(registry_entry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record_change(registry_entry_path, registry_entry_original)

    replay_source = root / "replay/replay_rr2.py"
    original_replay = replay_source.read_bytes()
    replay_text = original_replay.decode("utf-8").replace(
        "837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a",
        derivative_code_ledger_sha,
    )
    reviewer_parent_aggregate_sha = sha256(root / "upstream/forkaudit-summary.json")
    replay_text = replay_text.replace(
        "8700901ad7423d215e9e9e81a709e976f43963752e1b9f3d64441412b390d2bc",
        reviewer_parent_aggregate_sha,
    )
    replay_text = replay_text.replace(
        "import run_qcomem_qwen35_forkaudit_review_revision as runner",
        "import run_qcomem_qwen35_forkaudit_audit_protocol as runner",
    )
    derivative_source_count = len(code_rows)
    replay_text = replay_text.replace(
        'require(len(rows) == 34, f"expected 34 executed source files, found {len(rows)}")',
        f'require(len(rows) == {derivative_source_count}, '
        f'f"expected {derivative_source_count} reviewer-source files, found {{len(rows)}}")',
    )
    replay_text = replay_text.replace(
        '"executed_source_file_count_is_34": source_closure["source_file_count"] == 34,',
        '"reviewer_source_file_count_matches_derivative": '
        f'source_closure["source_file_count"] == {derivative_source_count},',
    )
    replay_source.write_text(replay_text, encoding="utf-8")
    record_change(replay_source, original_replay)

    # Rebind the eight governance-sanitized shard containers in both raw
    # authorities.  Tensor sidecars and all scientific values remain unchanged.
    raw_ledger = root / "upstream/receipts/all-raw-artifacts.sha256"
    original_raw_ledger = raw_ledger.read_bytes()
    raw_rows = []
    for line in original_raw_ledger.decode("utf-8").splitlines():
        digest, relative = line.split(None, 1)
        relative = relative.strip()
        path = root / "upstream" / relative
        if path in changed_raw_paths:
            digest = sha256(path)
        raw_rows.append(f"{digest}  {relative}")
    raw_ledger.write_text("\n".join(raw_rows) + "\n", encoding="utf-8")
    record_change(raw_ledger, original_raw_ledger)

    detached_path = root / "upstream/receipts/detached-receipt-manifest.json"
    original_detached = detached_path.read_bytes()
    detached = json.loads(original_detached)
    detached["protocol"] = "qcomem-qwen35-forkaudit-frozen-protocol-v1"
    raw_root = root / "upstream/raw"
    for row in detached["shards"]:
        path = raw_root / row["relative_path"]
        row["sha256"] = sha256(path)
        row["bytes"] = path.stat().st_size
    detached_path.write_text(
        json.dumps(detached, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record_change(detached_path, original_detached)

    notice = {
        "schema_version": "forkaudit-rr2-blind-derivative-scope-v2",
        "omitted_object": "author-side governance metadata",
        "reason": "blind package excludes author-side governance notes",
        "scientific_numeric_payloads_modified": False,
        "raw_shard_governance_fields_omitted": True,
        "reviewer_source_derivative_retains_scientific_method_bodies": True,
        "omitted_non_scientific_source_files": len(omitted_governance_sources),
        "omitted_original_executed_input_code_ledger": True,
        "reviewer_derivative_source_file_count": derivative_source_count,
        "replay_scope_unchanged": True,
    }
    (root / "REVIEWER_DERIVATIVE_SCOPE.json").write_text(
        json.dumps(notice, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # A few receipts undergo more than one governance-only rewrite (for
    # example, generic namespace neutralization followed by canonical receipt
    # rebinding).  Publish one original-to-final row per path so every declared
    # reviewer derivative hash is directly checkable against the frozen bytes.
    coalesced_rows: dict[str, dict[str, object]] = {}
    for row in transformation_rows:
        path = str(row["path"])
        previous = coalesced_rows.get(path)
        if previous is None:
            coalesced_rows[path] = dict(row)
            continue
        if previous["reviewer_derivative_sha256"] != row["original_sha256"]:
            raise RuntimeError(f"non-contiguous governance rewrite chain: {path}")
        if (
            "scientific_numeric_leaf_sha256" in previous
            and "scientific_numeric_leaf_sha256" in row
            and previous["scientific_numeric_leaf_sha256"]
            != row["scientific_numeric_leaf_sha256"]
        ):
            raise RuntimeError(f"numeric commitment drift across rewrites: {path}")
        combined = dict(row)
        combined["original_sha256"] = previous["original_sha256"]
        combined["transformation"] = (
            f"{previous['transformation']}; {row['transformation']}"
        )
        coalesced_rows[path] = combined

    (root / "BLIND_GOVERNANCE_REDACTION_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": "forkaudit-rr2-governance-redaction-v1",
                "numeric_leaf_commitment": {
                    "canonicalization": (
                        "sorted JSON-pointer/value rows after omitting hidden "
                        "governance nodes; booleans and source-count metadata excluded"
                    ),
                    "builder_asserts_original_equals_reviewer_derivative": True,
                    "original_preimages_withheld_for_blindness": True,
                },
                "rows": sorted(coalesced_rows.values(), key=lambda row: row["path"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "python3",
            str(replay_source),
            "--package-root",
            str(root),
            "--output",
            str(root / "derived"),
        ],
        check=True,
    )
    write_complete_package_manifest(
        root,
        schema_version="anonymous-hash-bound-reviewer-package-complete-v4",
        parent_manifest_sha256=(
            "51346e18c2d2685ea57712d1823e6056ea6bea11a5718da6d24f2fe1d1b65338"
        ),
    )


def refresh_lifecycle_manifest(root: Path) -> None:
    manifest_path = root / "replay/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file():
            if row["path"].startswith("artifacts/frozen-code/"):
                continue
            raise RuntimeError(f"lifecycle derivative member missing: {row['path']}")
        rows.append(
            {
                "path": row["path"],
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest["files"] = rows
    manifest["file_count"] = len(rows)
    manifest["total_bytes"] = sum(row["size_bytes"] for row in rows)
    manifest["schema_version"] = "forkaudit-lifecycle-blind-reviewer-derivative-v2"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "replay/MANIFEST.sha256").write_text(
        f"{sha256(manifest_path)}  MANIFEST.json\n", encoding="ascii"
    )


def prune_lifecycle_to_reviewer_manifest(root: Path) -> None:
    """Remove unmanifested author-side logs and receipts from the blind copy."""

    manifest = json.loads((root / "replay/MANIFEST.json").read_text(encoding="utf-8"))
    allowed = {row["path"] for row in manifest["files"]}
    allowed.update(
        {
            "replay/MANIFEST.json",
            "replay/MANIFEST.sha256",
            "BLIND_DERIVATIVE_SCOPE.json",
        }
    )
    for path in sorted(root.rglob("*"), reverse=True):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in allowed or relative.startswith("artifacts/frozen-code/"):
            continue
        path.chmod(path.stat().st_mode | 0o600)
        path.unlink()
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def redact_platform_fields(raw: bytes) -> bytes:
    derivative = raw
    key_pattern = re.compile(
        rb'(?P<prefix>["\']?(?:qs_job_id|qs_trial_id|job_id|trial_id|queue_id|cloud_id|cluster_id|resource_package_id|'
        rb'jobId|trialId|queueId|cloudId|clusterId|resourcePackageId)["\']?\s*:\s*)'
        rb'(?P<value>[0-9]+)'
    )
    derivative = key_pattern.sub(
        lambda match: match.group("prefix") + b'"<REDACTED_PLATFORM_ID>"', derivative
    )
    derivative = re.sub(
        rb'\b(?:Job|Trial)\s+[0-9]+\b', b"<REDACTED_PLATFORM_RECORD>", derivative
    )
    derivative = re.sub(
        rb'(/model/production/job/trial/)[0-9]+/[0-9]+',
        rb'\1<REDACTED_PLATFORM_ID>/<REDACTED_PLATFORM_ID>',
        derivative,
    )
    return derivative


PLATFORM_METADATA_KEYS = {
    "platform_receipt",
    "qs_job_id",
    "qs_trial_id",
    "job_id",
    "trial_id",
    "queue_id",
    "cloud_id",
    "cluster_id",
    "resource_package_id",
    "jobId",
    "trialId",
    "queueId",
    "cloudId",
    "clusterId",
    "resourcePackageId",
}


def scientific_numeric_leaf_sha256(raw: bytes, suffix: str) -> str | None:
    """Hash JSON numeric leaves after excluding scheduling-only namespaces."""

    if suffix == ".json":
        values = [json.loads(raw)]
    elif suffix == ".jsonl":
        values = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        return None
    rows: list[tuple[str, int | float]] = []

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key in sorted(node):
                if key in PLATFORM_METADATA_KEYS:
                    continue
                walk(node[key], f"{path}/{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}/{index}")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            rows.append((path, node))

    for index, value in enumerate(values):
        walk(value, f"/{index}")
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_fresh_gdn_reviewer_replay(
    gdn_root: Path, redaction_rows: list[dict[str, str]]
) -> None:
    replay_root = gdn_root / "reviewer_replay"
    replay_root.mkdir(parents=True, exist_ok=False)
    clone(
        gdn_root / "executed_source/qcomem_gdn_transition_oracle_reference_preregistered.py",
        replay_root / "reference.py",
    )
    clone(PAPER / "scripts/replay_fresh_gdn_derivative.py", replay_root / "replay.py")

    preregistration = gdn_root / "preregistration.json"
    reviewer_preregistration = replay_root / "preregistration.reviewer.json"
    reviewer_preregistration.write_bytes(preregistration.read_bytes())
    capture_source = gdn_root / "artifacts/raw/capture-manifest.json"
    capture = json.loads(capture_source.read_bytes())
    capture["preregistration_raw_sha256"] = sha256(reviewer_preregistration)
    for row in capture["rows"]:
        for receipt in row["arrays"].values():
            receipt["relative_path"] = "../artifacts/raw/" + receipt["relative_path"]
    reviewer_capture = replay_root / "capture-manifest.reviewer.json"
    reviewer_capture.write_text(
        json.dumps(capture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    by_path = {row["path"]: row for row in redaction_rows}
    prereg_relative = (
        "evidence/gdn_transition_oracle_preregistered_20260819d/preregistration.json"
    )
    capture_relative = (
        "evidence/gdn_transition_oracle_preregistered_20260819d/"
        "artifacts/raw/capture-manifest.json"
    )
    original_prereg_sha = by_path[prereg_relative]["original_sha256"]
    original_capture_sha = by_path[capture_relative]["original_sha256"]
    binding = {
        "schema_version": "forkaudit-execution-to-reviewer-derivative-v1",
        "execution_authority": {
            "preregistration_sha256": original_prereg_sha,
            "capture_manifest_sha256": original_capture_sha,
            "scientific_sidecars_modified": False,
        },
        "reviewer_derivative": {
            "preregistration_sha256": sha256(reviewer_preregistration),
            "capture_manifest_sha256": sha256(reviewer_capture),
            "transformation": (
                "identity/path/platform redaction plus capture binding and relative-path rewrite"
            ),
        },
    }
    binding_path = replay_root / "EXECUTION_TO_REVIEWER_BINDING.json"
    binding_path.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    inputs = [
        replay_root / "reference.py",
        replay_root / "replay.py",
        reviewer_preregistration,
        reviewer_capture,
        binding_path,
        gdn_root / "artifacts/oracle-result.json",
        *sorted((gdn_root / "artifacts/raw/sidecars").rglob("*.npy")),
    ]
    rows = [
        {
            "relative_path": path.relative_to(gdn_root).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in inputs
    ]
    if len(rows) != 46:
        raise RuntimeError(f"fresh GDN reviewer input cardinality drift: {len(rows)}")
    manifest = {
        "schema_version": "forkaudit-fresh-gdn-reviewer-inputs-v1",
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    (replay_root / "INPUTS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (replay_root / "run_replay.sh").write_text(
        """#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/fresh-gdn-reviewer.XXXXXX")
trap 'rm -rf "$TMP"' EXIT HUP INT TERM
export PYTHONDONTWRITEBYTECODE=1
python3 "$ROOT/replay.py" --root "$ROOT" --output "$TMP/result.json" --self-test
""",
        encoding="utf-8",
    )
    os.chmod(replay_root / "run_replay.sh", 0o755)


def build_a4_reviewer_replay(a4_root: Path) -> None:
    source_manifest = json.loads(
        (a4_root / "preregistration/source-manifest.json").read_text(encoding="utf-8")
    )
    source_root = a4_root / "executed_source"
    source_root.mkdir(parents=True, exist_ok=False)
    gpu_root = PAPER.parent / "gpu"
    for row in source_manifest["files"]:
        source = gpu_root / row["path"]
        if (
            not source.is_file()
            or source.stat().st_size != row["bytes"]
            or sha256(source) != row["sha256"]
        ):
            raise RuntimeError(f"A4 executed source drift: {row['path']}")
        clone(source, source_root / row["path"])

    clone(PAPER / "scripts/replay_a4_transformers_transfer.py", a4_root / "replay.py")
    inputs = [
        *sorted(source_root.iterdir()),
        a4_root / "replay.py",
        a4_root / "results/receipts/frozen-static-manifest.json",
        a4_root / "results/receipts/frozen-source-manifest.json",
        a4_root / "results/receipts/gpu-assignment.json",
        a4_root / "results/receipts/model-authority-pre.json",
        a4_root / "results/receipts/model-authority-terminal.json",
        a4_root / "results/forkaudit-transformers-transfer-aggregate.json",
        a4_root / "results/artifact-ledger.json",
        *sorted((a4_root / "results/raw/shards").glob("*.json")),
        *sorted((a4_root / "results/raw/logits").glob("*.bin")),
    ]
    rows = [
        {
            "relative_path": path.relative_to(a4_root).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in inputs
    ]
    if len(rows) != 31:
        raise RuntimeError(f"A4 reviewer input cardinality drift: {len(rows)}")
    manifest = {
        "schema_version": "forkaudit-transformers-reviewer-inputs-v1",
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    (a4_root / "INPUTS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (a4_root / "run_replay.sh").write_text(
        """#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
python3 "$ROOT/replay.py" --root "$ROOT" --self-test
""",
        encoding="utf-8",
    )
    os.chmod(a4_root / "run_replay.sh", 0o755)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round",
        type=int,
        default=5,
        dest="round_index",
        help="blind-review round to create (default: 5)",
    )
    args = parser.parse_args()
    if args.round_index < 0:
        raise ValueError("round index must be non-negative")
    round_root = PAPER / f"review/round_{args.round_index:02d}"
    target = round_root / "submission"
    staging = round_root / ".submission-staging"
    if target.exists() or staging.exists():
        raise RuntimeError(
            f"round-{args.round_index:02d} snapshot path already exists; refusing to overwrite"
        )
    staging.mkdir(parents=True)

    files = {
        "manuscript/main.pdf": PAPER / "main.pdf",
        "source/main.tex": PAPER / "main.tex",
        "source/references.bib": PAPER / "references.bib",
        "source/math_commands.tex": PAPER / "math_commands.tex",
        "source/iclr2026_conference.sty": PAPER / "iclr2026_conference.sty",
        "source/iclr2026_conference.bst": PAPER / "iclr2026_conference.bst",
        "source/fancyhdr.sty": PAPER / "fancyhdr.sty",
        "evidence/claim_evidence_map.tsv": PAPER / "evidence/claim_evidence_map.tsv",
        "evidence/method_provenance.tsv": PAPER / "evidence/method_provenance.tsv",
        "evidence/experiment_registry.json": PAPER / "evidence/experiment_registry.json",
        "evidence/seven_target_status.json": PAPER / "evidence/seven_target_status.json",
        "evidence/local_governance_tests.log": PAPER / "evidence/local_governance_tests.log",
        "evidence/integrated_results.json": PAPER / "evidence/integrated_results.json",
        "evidence/manuscript_evidence_audit.json": PAPER / "generated/manuscript_evidence_audit.json",
        "evidence/round_04_manifest_scope.md": PAPER / "evidence/round_04_manifest_scope.md",
        "evidence/rr4_detector_matrix/debug_attempt_d/all-debug-artifacts.sha256": PAPER / "evidence/rr4_detector_matrix/debug_attempt_d/all-debug-artifacts.sha256",
        "evidence/rr4_detector_matrix/debug_attempt_d/two-fault-local-comparison.json": PAPER / "evidence/rr4_detector_matrix/debug_attempt_d/two-fault-local-comparison.json",
        "evidence/rr4_detector_matrix/debug_attempt_d/w-run-m8-m9-raw-validation.json": PAPER / "evidence/rr4_detector_matrix/debug_attempt_d/w-run-m8-m9-raw-validation.json",
        "evidence/gdn_transition_oracle_capture_source.py": PAPER / "evidence/gdn_transition_oracle_capture_source.py",
        "scripts/validate_snapshot_provenance.py": PAPER / "scripts/validate_snapshot_provenance.py",
        "scripts/validate_snapshot_anonymity.py": PAPER / "scripts/validate_snapshot_anonymity.py",
        "scripts/generate_cross_hardware_tables.py": PAPER / "scripts/generate_cross_hardware_tables.py",
        "scripts/generate_h20_deployment_table.py": PAPER / "scripts/generate_h20_deployment_table.py",
        "scripts/generate_related_work_context_table.py": PAPER / "scripts/generate_related_work_context_table.py",
        "literature/citation_lock.json": PAPER / "literature/citation_lock.json",
        "literature/reported_system_context.json": PAPER / "literature/reported_system_context.json",
        "venue/review-rubric.md": Path.home() / ".codex/skills/autonomous-paper-agent-v2/references/review-rubric.md",
        "venue/iclr-kv-cache-calibration.md": Path.home() / ".codex/skills/autonomous-paper-agent-v2/references/iclr-kv-cache-calibration.md",
        "templates/review.schema.json": Path.home() / ".codex/skills/autonomous-paper-agent-v2/templates/review.schema.json",
    }
    directories = {
        "source/figures": PAPER / "figures",
        "source/tables": PAPER / "tables",
        "evidence/round_04_rr2_package": PAPER / "evidence/round_04_rr2_package",
        "evidence/gdn_transition_oracle_replay": PAPER / "evidence/gdn_transition_oracle_replay",
        "evidence/gdn_transition_oracle_preregistered_20260819d": PAPER / "evidence/gdn_transition_oracle_preregistered_20260819d",
        "evidence/lifecycle_transfer_reviewer_package": PAPER / "evidence/lifecycle_transfer_reviewer_package",
        "evidence/round6_a4_transformers_transfer_20260819b": PAPER / "evidence/round6_a4_transformers_transfer_20260819b",
        "evidence/mac_m4_motivation": PAPER / "evidence/mac_m4_motivation",
        "evidence/h20_deployment_benchmark": PAPER / "evidence/h20_deployment_benchmark",
        "supplement_anonymous": PAPER / "supplement_anonymous",
    }
    for relative, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        clone(source, staging / relative)
    for relative, source in directories.items():
        if not source.is_dir():
            raise FileNotFoundError(source)
        clone(source, staging / relative)

    lifecycle_root = staging / "evidence/lifecycle_transfer_reviewer_package"
    lifecycle_frozen_code = lifecycle_root / "artifacts/frozen-code"
    if not lifecycle_frozen_code.is_dir():
        raise RuntimeError("lifecycle frozen-code closure missing before blind omission")
    for path in sorted(lifecycle_frozen_code.rglob("*"), reverse=True):
        path.chmod(path.stat().st_mode | (0o700 if path.is_dir() else 0o600))
    lifecycle_frozen_code.chmod(lifecycle_frozen_code.stat().st_mode | 0o700)
    shutil.rmtree(lifecycle_frozen_code)
    (lifecycle_root / "BLIND_DERIVATIVE_SCOPE.json").write_text(
        json.dumps(
            {
                "schema_version": "forkaudit-lifecycle-blind-scope-v1",
                "omitted_object": "redundant author-side GPU source closure",
                "scientific_raw_shards_modified": False,
                "replay_validator_retained": True,
                "unmanifested_author_logs_and_receipts_omitted": True,
                "opaque_execution_code_digest_retained": True,
                "original_source_filename_inventory_retained": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    prune_lifecycle_to_reviewer_manifest(lifecycle_root)

    prepare_rr2_blind_derivative(staging / "evidence/round_04_rr2_package")
    old_runner_path = b"run_qcomem_qwen35_forkaudit_review_revision.py"
    new_runner_path = b"run_qcomem_qwen35_forkaudit_audit_protocol.py"
    for relative in (
        "evidence/method_provenance.tsv",
        "evidence/seven_target_status.json",
    ):
        path = staging / relative
        original = path.read_bytes()
        if old_runner_path not in original:
            raise RuntimeError(f"RR2 reviewer locator missing before rewrite: {relative}")
        path.write_bytes(original.replace(old_runner_path, new_runner_path))
    refresh_rr2_snapshot_metadata(staging)

    citation_lock_path = staging / "literature/citation_lock.json"
    citation_lock = json.loads(citation_lock_path.read_text(encoding="utf-8"))
    for entry in citation_lock.get("entries", []):
        if entry.get("key") == "liu2026comem":
            entry["primary_url"] = "<REDACTED_FOR_DOUBLE_BLIND>"
    citation_lock_path.write_text(
        json.dumps(citation_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # The fresh A2/A4 evidence is copied from the immutable run packages.  A
    # small subset of its textual receipts contains author-local mount, job,
    # or user strings.  Redact only those strings in the blind derivative;
    # numerical sidecars remain byte-identical.  Preserve both original and
    # derivative hashes so reviewers can distinguish execution authority from
    # anonymized packaging.
    run_prefix = (
        b"/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/"
        b"indep-bench_assets/runs/qcomem/"
        b"qwen35-gdn-transition-oracle-preregistered-20260819e"
    )
    replacements = (
        (run_prefix, b"."),
        (b"https://qs2.devops.xiaohongshu.com", b"<REDACTED_PLATFORM>"),
        (b"artifactory.devops.xiaohongshu.com", b"<REDACTED_IMAGE_REGISTRY>"),
        (b"/Users/liuhanzuo", b"<REDACTED_LOCAL_HOME>"),
        (
            b"/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo",
            b"<REDACTED_SHARED_ROOT>",
        ),
        (b"/mnt/tidal-alsh-hilab", b"<REDACTED_MOUNT>"),
        (b"liuhanzuo", b"<REDACTED_USER>"),
        (b"Reviewer impact", b"Evidence impact"),
        (b"A-EXPERIMENT-07", b"BOUNDED-LIFECYCLE-COHORT"),
    )
    redaction_roots = (
        staging / "evidence/gdn_transition_oracle_replay",
        staging / "evidence/gdn_transition_oracle_preregistered_20260819d",
        staging / "evidence/round6_a4_transformers_transfer_20260819b",
    )
    redaction_rows = []
    textual_suffixes = {
        "", ".json", ".jsonl", ".yaml", ".yml", ".md", ".txt", ".log",
        ".sha256", ".py", ".sh",
    }
    # Remove author-side scheduling identifiers before computing the exact
    # reviewer-derivative hashes recorded below.  No later rewrite may make a
    # redaction-manifest row stale.
    registry_path = staging / "evidence/experiment_registry.json"
    registry_original = registry_path.read_bytes()
    registry = json.loads(registry_original)
    for experiment in registry.get("experiments", []):
        experiment.pop("platform_receipt", None)
        experiment.pop("run_id", None)
        experiment["artifact_paths"] = [
            path.replace(
                old_runner_path.decode("ascii"), new_runner_path.decode("ascii")
            )
            for path in experiment.get("artifact_paths", [])
        ]
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    redaction_rows.append(
        {
            "path": registry_path.relative_to(staging).as_posix(),
            "original_sha256": hashlib.sha256(registry_original).hexdigest(),
            "reviewer_derivative_sha256": sha256(registry_path),
            "scientific_numeric_leaf_sha256": scientific_numeric_leaf_sha256(
                registry_original, ".json"
            ),
            "transformation": (
                "remove author-side scheduling identifiers and bind the neutral "
                "reviewer-source runner locator"
            ),
        }
    )

    redaction_candidates = [
        path
        for root in redaction_roots
        for path in sorted(root.rglob("*"))
    ] + [
        staging / "evidence/lifecycle_transfer_reviewer_package/result_report.md",
        staging / "evidence/lifecycle_transfer_reviewer_package/validation_report.json",
        staging / "evidence/lifecycle_transfer_reviewer_package/design_decision.md",
        staging / "evidence/rr4_detector_matrix/debug_attempt_d/two-fault-local-comparison.json",
        staging / "evidence/rr4_detector_matrix/debug_attempt_d/w-run-m8-m9-raw-validation.json",
    ]
    for path in redaction_candidates:
        if not path.is_file() or path.suffix.lower() not in textual_suffixes:
            continue
        original = path.read_bytes()
        derivative = original
        for old, new in replacements:
            derivative = derivative.replace(old, new)
        derivative = redact_platform_fields(derivative)
        if derivative == original:
            continue
        original_numeric = scientific_numeric_leaf_sha256(original, path.suffix.lower())
        derivative_numeric = scientific_numeric_leaf_sha256(
            derivative, path.suffix.lower()
        )
        if original_numeric != derivative_numeric:
            raise RuntimeError(
                f"scientific numeric leaves changed during blind redaction: {path}"
            )
        path.write_bytes(derivative)
        row = {
            "path": path.relative_to(staging).as_posix(),
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "reviewer_derivative_sha256": hashlib.sha256(derivative).hexdigest(),
            "transformation": "fixed-string identity/path redaction only",
        }
        if derivative_numeric is not None:
            row["scientific_numeric_leaf_sha256"] = derivative_numeric
        redaction_rows.append(row)

    # Execution hashes remain useful provenance, but after redaction they must
    # not masquerade as hashes of reviewer-visible bytes.  Preserve the former
    # under explicitly qualified keys and bind the actual blind derivatives.
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    gdn_experiment = next(
        experiment
        for experiment in registry["experiments"]
        if experiment["evidence_id"] == "E-R6-FULLY-PREREGISTERED-GDN-ORACLE"
    )
    execution_prereg = gdn_experiment.pop("preregistration_sha256")
    execution_config = gdn_experiment.pop("submitted_config_sha256")
    reviewer_prereg_path = (
        staging
        / "evidence/gdn_transition_oracle_preregistered_20260819d/preregistration.json"
    )
    reviewer_config_path = (
        staging
        / "evidence/gdn_transition_oracle_preregistered_20260819d/"
        "submitted-config-20260819e.yaml"
    )
    gdn_experiment["execution_preregistration_sha256"] = execution_prereg
    gdn_experiment["reviewer_preregistration_sha256"] = sha256(reviewer_prereg_path)
    gdn_experiment["execution_submitted_config_sha256"] = execution_config
    gdn_experiment["reviewer_submitted_config_sha256"] = sha256(
        reviewer_config_path
    )
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry_row = next(
        row
        for row in redaction_rows
        if row["path"] == "evidence/experiment_registry.json"
    )
    registry_row["reviewer_derivative_sha256"] = sha256(registry_path)
    registry_row["transformation"] = (
        "remove author-side scheduling identifiers; bind the neutral reviewer "
        "runner locator; qualify execution hashes and bind reviewer-visible "
        "derivative bytes"
    )

    refresh_plain_package_manifest(staging / "evidence/gdn_transition_oracle_replay")
    refresh_lifecycle_manifest(
        staging / "evidence/lifecycle_transfer_reviewer_package"
    )
    redaction_manifest = {
        "schema_version": "forkaudit-blind-redaction-manifest-v1",
        "numerical_binary_sidecars_modified": False,
        "numeric_leaf_commitment": {
            "canonicalization": (
                "sorted JSON-pointer/value rows; scheduling-only namespaces "
                "and booleans excluded"
            ),
            "builder_asserts_original_equals_reviewer_derivative": True,
            "original_preimages_withheld_for_blindness": True,
        },
        "rows": redaction_rows,
    }
    (staging / "evidence/BLIND_REDACTION_MANIFEST.json").write_text(
        json.dumps(redaction_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_fresh_gdn_reviewer_replay(
        staging / "evidence/gdn_transition_oracle_preregistered_20260819d",
        redaction_rows,
    )
    build_a4_reviewer_replay(
        staging / "evidence/round6_a4_transformers_transfer_20260819b"
    )

    readme = """# ForkAudit blind reviewer snapshot

This snapshot contains no previous reviews, scores, author-side governance
notes, or target rating.  It includes the anonymous RR2 replay package and all
static manuscript build inputs.

## Reproduce the primary offline audit

```bash
cd evidence/round_04_rr2_package
./replay/run_replay.sh
```

The replay verifies its own raw-byte manifest before reconstructing factorial,
timeline, storage, attention-oracle, mutation, and memory results.  It requires
CPU-only Python plus NumPy and does not regenerate the live GPU execution.

## Reproduce the fresh fully preregistered GDN rows

```bash
cd evidence/gdn_transition_oracle_preregistered_20260819d/reviewer_replay
./run_replay.sh
```

This manifest-first replay verifies the 40 original numerical sidecars,
recomputes four clean recurrent transitions, confirms rejection of four seeded
wrong transitions, and requires exact producer/reviewer decisions with
explicitly bounded portable FP32 metric deltas.  Its binding file records both the immutable execution hashes
and the anonymized reviewer-derivative hashes.  The final preregistration and
source pin precede this execution; no amendment is used.

## Inspect the same-model Transformers-runtime transfer

```bash
cd evidence/round6_a4_transformers_transfer_20260819b
./run_replay.sh
```

This replays all eight raw shards and eight CPU-FP32 logit bundles through the
exact seven-file executed source closure and requires exact equality with the
producer aggregate.  `results/` contains all
eight raw shards, eight CPU-FP32 logit bundles, the formal aggregate, and the
24-row terminal artifact ledger.  The run is a valid negative: ownership and
relational targets hold, but all eight matched-clean ranks fail the frozen
dense semantic oracle.  It is not positive runtime-portability evidence.

## Reproduce the lifecycle transfer

```bash
cd evidence/lifecycle_transfer_reviewer_package
./replay/run_replay.sh --self-test
```

This manifest-first validator reconstructs the eight raw-shard predicates and
rejects raw-byte, stale-lease, scrub, and reservation-reassignment tampering.

## Audit snapshot-local provenance

```bash
python3 scripts/validate_snapshot_provenance.py .
python3 scripts/validate_snapshot_anonymity.py .
```

The strict audit verifies every snapshot-manifest member, resolves every
claim evidence ID through the global registry, resolves every method path and
Python symbol, and checks the seven-target status vector.

## Rebuild the manuscript

```bash
cd source
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`evidence/lifecycle_transfer_reviewer_package/` contains the frozen lifecycle
artifacts, raw-first replay, and independent validation report.
`evidence/integrated_results.json`
records the bounded GDN-oracle and lifecycle results and the explicit
missingness of the nonformal M8/M9 debug.  The primary
GPU aggregate and offline replay are separate verified objects; no single-run
terminal-closure claim is made.

`evidence/BLIND_REDACTION_MANIFEST.json` records every textual receipt changed
only to remove private identity, mount, or platform strings.  Numerical binary
sidecars are copied byte-for-byte; original hashes remain in the execution
receipts and derivative hashes are bound by the top-level snapshot manifest.
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")

    subprocess.run(
        [
            "python3",
            str(staging / "scripts/validate_snapshot_anonymity.py"),
            str(staging),
            "--output",
            "evidence/BLIND_ANONYMITY_AUDIT.json",
        ],
        check=True,
    )

    records = []
    total_bytes = 0
    for path in sorted(staging.rglob("*")):
        # Exclude only the top-level manifest being constructed.  Nested
        # package manifests are ordinary evidence and must themselves be
        # bound by the snapshot identity.
        if not path.is_file() or path == staging / "MANIFEST.json":
            continue
        relative = path.relative_to(staging).as_posix()
        size = path.stat().st_size
        records.append({"path": relative, "sha256": sha256(path), "size_bytes": size})
        total_bytes += size
    identity = hashlib.sha256()
    for row in records:
        identity.update(row["path"].encode("utf-8"))
        identity.update(b"\0")
        identity.update(row["sha256"].encode("ascii"))
        identity.update(b"\n")
    manifest = {
        "schema_version": "1.0.0",
        "round": args.round_index,
        "venue": "ICLR 2026",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "blind_snapshot": True,
        "snapshot_sha256": identity.hexdigest(),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "manuscript_pdf_sha256": sha256(staging / "manuscript/main.pdf"),
        "manuscript_source_sha256": sha256(staging / "source/main.tex"),
        "rr2_package_manifest_sha256": sha256(
            staging / "evidence/round_04_rr2_package/MANIFEST.json"
        ),
        "gdn_replay_manifest_sha256": sha256(
            staging / "evidence/gdn_transition_oracle_replay/MANIFEST.json"
        ),
        "fully_preregistered_gdn_validation_sha256": sha256(
            staging / "evidence/gdn_transition_oracle_preregistered_20260819d/validation_report.json"
        ),
        "fully_preregistered_gdn_reviewer_inputs_sha256": sha256(
            staging
            / "evidence/gdn_transition_oracle_preregistered_20260819d"
            / "reviewer_replay/INPUTS_MANIFEST.json"
        ),
        "lifecycle_replay_manifest_sha256": sha256(
            staging / "evidence/lifecycle_transfer_reviewer_package/replay/MANIFEST.json"
        ),
        "transformers_transfer_aggregate_sha256": sha256(
            staging / "evidence/round6_a4_transformers_transfer_20260819b/results/forkaudit-transformers-transfer-aggregate.json"
        ),
        "transformers_transfer_artifact_ledger_sha256": sha256(
            staging / "evidence/round6_a4_transformers_transfer_20260819b/results/artifact-ledger.json"
        ),
        "transformers_transfer_reviewer_inputs_sha256": sha256(
            staging
            / "evidence/round6_a4_transformers_transfer_20260819b"
            / "INPUTS_MANIFEST.json"
        ),
        "files": records,
        "excluded": [
            "previous reviews and scores",
            "author response plans and issue ledger",
            "target rating",
            "private platform identifiers and host paths",
            "invalid or superseded experiment attempts"
        ]
    }
    (staging / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.replace(target)
    print(json.dumps({
        "status": "passed",
        "snapshot": str(target),
        "snapshot_sha256": manifest["snapshot_sha256"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"]
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
