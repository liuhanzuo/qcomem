#!/usr/bin/env python3
"""Reconcile the 135-entry execution ledger with the released source subset.

This Round-02 RP04 audit intentionally emits no original absolute paths.  It
binds the original ledger by SHA-256, normalizes every row to a unique basename,
matches exact historical bytes, computes local static-import closures, and
checks an import-only runtime module trace when the local Python environment can
load the frozen anonymous source copy.

The audit is fail-closed: malformed or drifted provenance inputs abort without
replacing the report.  Source rows that cannot be classified from exact bytes
are retained as ``unresolved``; they are never assigned a guessed role.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PAPER = Path(__file__).resolve().parents[1]
REPOSITORY = PAPER.parent
PRIMARY = REPOSITORY / "results/gpu-qwen35-vllm-paged-multifork-resident-20260814a"
ANONYMOUS = PAPER / "supplement_anonymous"
ROUND02_SNAPSHOT = PAPER / "review/round_02/submission"
ROUND02_ANONYMOUS = ROUND02_SNAPSHOT / "submission/supplement_anonymous"
OUTPUT = PAPER / "evidence/execution_closure_reconciliation.json"

ORIGINAL_LEDGER = PRIMARY / "code.sha256"
ANONYMOUS_LEDGER = ANONYMOUS / "code/EXECUTED_SOURCE_SHA256"
ORIGINAL_EVIDENCE = ANONYMOUS / "provenance/original_evidence_digests.json"
ANONYMOUS_MANIFEST = ANONYMOUS / "ANONYMOUS_MANIFEST.sha256"
ROUND02_MANIFEST = ROUND02_SNAPSHOT / "MANIFEST.json"
ROUND02_ANONYMOUS_LEDGER = ROUND02_ANONYMOUS / "code/EXECUTED_SOURCE_SHA256"
ROUND02_ANONYMOUS_MANIFEST = ROUND02_ANONYMOUS / "ANONYMOUS_MANIFEST.sha256"

EXPECTED_ORIGINAL_LEDGER_SHA256 = (
    "44c3a86a2cd7db7afcb7ee0cb29af91625ec3fcf5374c509146d92a451824ff9"
)
EXPECTED_ORIGINAL_ENTRY_COUNT = 135
EXPECTED_ANONYMOUS_LEDGER_SHA256 = (
    "ca96cc2af4d03f131eaffe80713a5ce20f7474ebff31d2a8624fdbb5a21aa7c2"
)
EXPECTED_ANONYMOUS_ENTRY_COUNT = 18
EXPECTED_ROUND02_SNAPSHOT_SHA256 = (
    "4aabda8d5466e6456b5e66f8717a725f66723d326a37626ff09890f68d192d32"
)

RUNTIME_ROOT = "run_qcomem_qwen35_vllm_paged_multifork_resident.py"
LAUNCHER = "launch_qcomem_qwen35_vllm_paged_multifork_resident_8gpu.sh"
PROTOCOL = "MULTIFORK_RESIDENT_PROTOCOL_ZH.md"
PROTOCOL_OR_LAUNCHER = {LAUNCHER, PROTOCOL}

LEDGER_LINE = re.compile(r"([0-9a-f]{64})  (.+)\Z")
RELATIVE_MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  ([^\n]+)\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SENSITIVE_OUTPUT = (
    re.compile(r"/Users/", re.IGNORECASE),
    re.compile(r"/private/", re.IGNORECASE),
    re.compile(r"/mnt/", re.IGNORECASE),
    re.compile(r"liuhanzuo", re.IGNORECASE),
    re.compile(r"tidal-alsh-hilab", re.IGNORECASE),
)


class ReconciliationError(RuntimeError):
    """Raised when a frozen provenance invariant does not hold."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconciliationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_ledger_sha256(rows: Iterable[tuple[str, str]]) -> str:
    payload = "".join(
        f"{digest}  {logical_name}\n"
        for logical_name, digest in sorted(rows, key=lambda item: item[0])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_original_ledger(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    original_parents: set[str] = set()
    for position, line in enumerate(lines, start=1):
        match = LEDGER_LINE.fullmatch(line)
        require(match is not None, f"original ledger line {position} is malformed")
        digest, original_path = match.groups()
        parsed = PurePosixPath(original_path)
        require(parsed.is_absolute(), f"original ledger line {position} is not absolute")
        require(parsed.name not in {"", ".", ".."}, f"invalid logical name at line {position}")
        original_parents.add(parsed.parent.as_posix())
        rows.append(
            {
                "ledger_position": position,
                "logical_name": parsed.name,
                "sha256": digest,
            }
        )
    require(len(original_parents) == 1, "original ledger is not one flat source snapshot")
    return rows


def normalized_relative_name(value: str, label: str) -> str:
    parsed = PurePosixPath(value)
    require(not parsed.is_absolute(), f"{label} contains an absolute path")
    require(".." not in parsed.parts, f"{label} contains parent traversal")
    normalized = parsed.as_posix()
    require(normalized not in {"", "."}, f"{label} contains an empty path")
    return normalized


def parse_relative_manifest(path: Path, label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for position, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = RELATIVE_MANIFEST_LINE.fullmatch(line)
        require(match is not None, f"{label} line {position} is malformed")
        digest, relative = match.groups()
        rows.append(
            {
                "sha256": digest,
                "relative_path": normalized_relative_name(relative, label),
            }
        )
    names = [row["relative_path"] for row in rows]
    require(len(names) == len(set(names)), f"{label} contains duplicate paths")
    return rows


def verify_relative_manifest(root: Path, rows: list[dict[str, str]], label: str) -> None:
    resolved_root = root.resolve()
    for row in rows:
        target = (root / row["relative_path"]).resolve()
        require(
            target == resolved_root or resolved_root in target.parents,
            f"{label} escapes its root",
        )
        require(target.is_file(), f"{label} references a missing file")
        require(sha256_file(target) == row["sha256"], f"{label} digest mismatch")


def snapshot_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: str(row["snapshot_path"])):
        digest.update(str(record["snapshot_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_round02_snapshot() -> tuple[dict[str, Any], list[dict[str, str]]]:
    payload = json.loads(ROUND02_MANIFEST.read_text(encoding="utf-8"))
    require(payload.get("round") == 2, "review snapshot is not round 2")
    records = payload.get("files")
    require(isinstance(records, list) and records, "review snapshot has no file records")
    require(
        payload.get("snapshot_sha256") == EXPECTED_ROUND02_SNAPSHOT_SHA256,
        "round-02 snapshot identity drift",
    )
    require(snapshot_digest(records) == payload["snapshot_sha256"], "snapshot record digest mismatch")

    seen: set[str] = set()
    snapshot_root = ROUND02_SNAPSHOT.resolve()
    for record in records:
        require(isinstance(record, dict), "snapshot file record is not an object")
        relative = normalized_relative_name(str(record.get("snapshot_path", "")), "snapshot manifest")
        require(relative not in seen, "snapshot manifest contains a duplicate path")
        seen.add(relative)
        digest = record.get("sha256")
        size = record.get("size_bytes")
        require(isinstance(digest, str) and SHA256.fullmatch(digest), "invalid snapshot digest")
        require(isinstance(size, int) and size >= 0, "invalid snapshot byte count")
        target = (ROUND02_SNAPSHOT / relative).resolve()
        require(snapshot_root in target.parents, "snapshot path escapes snapshot root")
        require(target.is_file(), "snapshot manifest references a missing file")
        require(target.stat().st_size == size, "snapshot file size drift")
        require(sha256_file(target) == digest, "snapshot file digest drift")

    embedded_rows = parse_relative_manifest(
        ROUND02_ANONYMOUS_MANIFEST, "round-02 anonymous manifest"
    )
    verify_relative_manifest(
        ROUND02_ANONYMOUS, embedded_rows, "round-02 anonymous manifest"
    )
    return payload, embedded_rows


def exact_source_catalog(
    original_rows: list[dict[str, Any]],
) -> tuple[dict[str, Path], dict[str, list[str]], list[str]]:
    exact_sources: dict[str, Path] = {}
    matches: dict[str, list[str]] = {}
    worktree_drift: list[str] = []
    candidates = (
        ("anonymous_supplement_frozen_copy", ANONYMOUS / "code"),
        ("round02_review_snapshot_frozen_copy", ROUND02_ANONYMOUS / "code"),
        ("repository_digest_matched_copy", REPOSITORY / "gpu"),
    )
    for row in original_rows:
        name = row["logical_name"]
        expected = row["sha256"]
        row_matches: list[str] = []
        for label, root in candidates:
            candidate = root / name
            if not candidate.is_file():
                continue
            if sha256_file(candidate) == expected:
                row_matches.append(label)
                if name not in exact_sources:
                    exact_sources[name] = candidate
            elif label == "repository_digest_matched_copy":
                worktree_drift.append(name)
        matches[name] = sorted(row_matches)
    return exact_sources, matches, sorted(set(worktree_drift))


def dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def analyze_python_imports(
    sources: dict[str, Path], module_to_file: dict[str, str]
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    edges: dict[str, set[str]] = {name: set() for name in sources}
    unresolved: dict[str, list[str]] = {}
    for name, path in sorted(sources.items()):
        if path.suffix != ".py":
            continue
        issues: list[str] = []
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        except (OSError, UnicodeDecodeError, SyntaxError):
            unresolved[name] = ["exact source bytes could not be parsed as Python"]
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in module_to_file:
                        edges[name].add(module_to_file[root])
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    issues.append(f"relative import at line {node.lineno}")
                    continue
                if node.module:
                    root = node.module.split(".", 1)[0]
                    if root in module_to_file:
                        edges[name].add(module_to_file[root])
            elif isinstance(node, ast.Call):
                call_name = dotted_name(node.func)
                if call_name in {"__import__", "importlib.import_module", "import_module"}:
                    if not node.args or not isinstance(node.args[0], ast.Constant):
                        issues.append(f"non-literal dynamic import at line {node.lineno}")
                    elif isinstance(node.args[0].value, str):
                        root = node.args[0].value.split(".", 1)[0]
                        if root in module_to_file:
                            edges[name].add(module_to_file[root])
                elif call_name in {"exec", "eval"}:
                    issues.append(f"dynamic code execution at line {node.lineno}")
        if issues:
            unresolved[name] = sorted(set(issues))
    return edges, unresolved


def transitive_closure(roots: Iterable[str], edges: dict[str, set[str]]) -> set[str]:
    pending = list(roots)
    reached: set[str] = set()
    while pending:
        name = pending.pop()
        require(name in edges, f"closure root has no exact parsed source: {name}")
        if name in reached:
            continue
        reached.add(name)
        pending.extend(sorted(edges[name] - reached, reverse=True))
    return reached


def infer_launcher_roots(launcher_path: Path) -> tuple[str, list[str]]:
    text = launcher_path.read_text(encoding="utf-8")
    collapsed = re.sub(r"\\\n\s*", " ", text)
    test_match = re.search(
        r"\$ENV_DIR/bin/python\"?\s+-m\s+unittest\s+-v\s+(.*?)"
        r"\s+>\s*\"\$RUN_DIR/logs/focused-tests\.log\"",
        collapsed,
    )
    require(test_match is not None, "cannot infer focused-test roots from frozen launcher")
    test_modules = re.findall(r"\btest_[A-Za-z0-9_]+\b", test_match.group(1))
    require(test_modules, "frozen launcher has no focused-test roots")
    require(len(test_modules) == len(set(test_modules)), "focused-test roots are duplicated")
    test_files = [f"{module}.py" for module in test_modules]

    runtime_files = set(re.findall(r"\$CODE_DIR/(run_[A-Za-z0-9_]+\.py)", text))
    require(runtime_files == {RUNTIME_ROOT}, "frozen launcher has an unexpected runtime root")
    return RUNTIME_ROOT, test_files


def infer_anonymous_governance_roots() -> list[str]:
    log = (ANONYMOUS / "provenance/governance_tests.log").read_text(encoding="utf-8")
    first_line = log.splitlines()[0] if log.splitlines() else ""
    require(first_line.startswith("Command: python -m unittest -v "), "governance log command missing")
    modules = re.findall(r"\btest_[A-Za-z0-9_]+\b", first_line)
    require(modules and len(modules) == len(set(modules)), "invalid anonymous governance roots")
    return [f"{module}.py" for module in modules]


def run_import_only_trace(
    runtime_closure: set[str], module_to_file: dict[str, str]
) -> dict[str, Any]:
    module_names = sorted(module_to_file)
    child = f"""
import importlib
import json
import sys
expected = set({json.dumps(module_names)})
importlib.import_module({RUNTIME_ROOT[:-3]!r})
observed = sorted(name for name in expected if name in sys.modules)
print(json.dumps(observed, separators=(\",\", \":\")))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, "-c", child],
        cwd=ANONYMOUS / "code",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if process.returncode != 0:
        return {
            "status": "unavailable",
            "reason": "the active local Python environment could not import the frozen root",
            "gpu_execution_attempted": False,
        }
    try:
        observed_modules = json.loads(process.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        raise ReconciliationError("runtime import trace did not emit a valid module list")
    require(isinstance(observed_modules, list), "runtime import trace is not a list")
    observed_files = sorted(module_to_file[module] for module in observed_modules)
    require(
        set(observed_files) == runtime_closure,
        "runtime import trace disagrees with static local-import closure",
    )
    return {
        "status": "completed",
        "method": "fresh-process import-only trace of the exact anonymous frozen runtime root",
        "root": RUNTIME_ROOT,
        "observed_local_module_count": len(observed_files),
        "observed_local_modules": observed_files,
        "equals_static_runtime_closure": True,
        "gpu_execution_attempted": False,
        "limitation": (
            "This is a module-import trace, not a GPU execution trace; conditional imports "
            "inside functions are covered only by the static analysis."
        ),
    }


def classification_basis(role: str) -> str:
    return {
        "runtime_reachable": (
            "statically reachable from the launcher-inferred runtime root using exact "
            "digest-matched source bytes"
        ),
        "test_only": (
            "statically reachable from launcher-inferred focused-test roots but not from "
            "the runtime root"
        ),
        "protocol_or_launcher": "explicit non-Python launcher or protocol ledger member",
        "other_frozen_dependency": (
            "frozen ledger member outside the known runtime and focused-test closures; "
            "this label asserts snapshot membership, not target-runtime reachability"
        ),
        "unresolved": "exact source bytes or a safe static classification were unavailable",
    }[role]


def assert_privacy_safe(value: Any, pointer: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_privacy_safe(key, f"{pointer}.<key>")
            assert_privacy_safe(item, f"{pointer}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_privacy_safe(item, f"{pointer}[{index}]")
    elif isinstance(value, str):
        for pattern in SENSITIVE_OUTPUT:
            require(pattern.search(value) is None, f"privacy-sensitive value at {pointer}")
        require(not value.startswith("/"), f"absolute path at {pointer}")
        require("file://" not in value.lower(), f"file URI at {pointer}")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    require(sha256_file(ORIGINAL_LEDGER) == EXPECTED_ORIGINAL_LEDGER_SHA256, "ledger identity drift")
    require(
        sha256_file(ANONYMOUS_LEDGER) == EXPECTED_ANONYMOUS_LEDGER_SHA256,
        "anonymous source ledger identity drift",
    )
    require(
        sha256_file(ROUND02_ANONYMOUS_LEDGER) == EXPECTED_ANONYMOUS_LEDGER_SHA256,
        "round-02 anonymous source ledger identity drift",
    )

    original_rows = parse_original_ledger(ORIGINAL_LEDGER)
    require(len(original_rows) == EXPECTED_ORIGINAL_ENTRY_COUNT, "original ledger count drift")
    logical_names = [row["logical_name"] for row in original_rows]
    digests = [row["sha256"] for row in original_rows]
    require(len(set(logical_names)) == EXPECTED_ORIGINAL_ENTRY_COUNT, "duplicate normalized names")
    require(len(set(digests)) == EXPECTED_ORIGINAL_ENTRY_COUNT, "original ledger has duplicate digests")

    anonymous_rows_raw = parse_relative_manifest(ANONYMOUS_LEDGER, "anonymous source ledger")
    anonymous_rows = [
        {
            "logical_name": PurePosixPath(row["relative_path"]).name,
            "sha256": row["sha256"],
        }
        for row in anonymous_rows_raw
    ]
    require(len(anonymous_rows) == EXPECTED_ANONYMOUS_ENTRY_COUNT, "anonymous ledger count drift")
    anonymous_names = [row["logical_name"] for row in anonymous_rows]
    anonymous_digests = [row["sha256"] for row in anonymous_rows]
    require(len(set(anonymous_names)) == len(anonymous_names), "anonymous logical names repeat")
    require(len(set(anonymous_digests)) == len(anonymous_digests), "anonymous digests repeat")

    original_by_name = {row["logical_name"]: row["sha256"] for row in original_rows}
    anonymous_by_name = {row["logical_name"]: row["sha256"] for row in anonymous_rows}
    anonymous_unmatched = sorted(
        name for name, digest in anonymous_by_name.items() if original_by_name.get(name) != digest
    )
    require(not anonymous_unmatched, "anonymous source ledger is not an exact subset")
    require(set(anonymous_by_name) < set(original_by_name), "anonymous source ledger is not strict subset")

    original_evidence = json.loads(ORIGINAL_EVIDENCE.read_text(encoding="utf-8"))
    bound_ledger = original_evidence.get("original_execution_code_ledger", {})
    require(bound_ledger.get("sha256") == EXPECTED_ORIGINAL_LEDGER_SHA256, "evidence ledger hash drift")
    require(bound_ledger.get("entry_count") == EXPECTED_ORIGINAL_ENTRY_COUNT, "evidence count drift")
    require(bound_ledger.get("bytes") == ORIGINAL_LEDGER.stat().st_size, "evidence byte count drift")
    evidence_code = {
        PurePosixPath(row["file"]).name: row["sha256"]
        for row in original_evidence.get("code", [])
    }
    require(evidence_code == anonymous_by_name, "published code digest records disagree with subset")

    live_manifest_rows = parse_relative_manifest(ANONYMOUS_MANIFEST, "anonymous manifest")
    verify_relative_manifest(ANONYMOUS, live_manifest_rows, "anonymous manifest")
    round02_payload, round02_manifest_rows = verify_round02_snapshot()
    snapshot_anon_rows = parse_relative_manifest(
        ROUND02_ANONYMOUS_LEDGER, "round-02 anonymous source ledger"
    )
    snapshot_anon_by_name = {
        PurePosixPath(row["relative_path"]).name: row["sha256"] for row in snapshot_anon_rows
    }
    require(snapshot_anon_by_name == anonymous_by_name, "live and round-02 code subsets differ")

    exact_sources, source_matches, worktree_drift = exact_source_catalog(original_rows)
    python_names = sorted(name for name in logical_names if name.endswith(".py"))
    module_to_file = {Path(name).stem: name for name in python_names}
    require(len(module_to_file) == len(python_names), "Python module stem collision")
    python_sources = {name: exact_sources[name] for name in python_names if name in exact_sources}
    edges, static_issues = analyze_python_imports(python_sources, module_to_file)
    for name in python_names:
        if name not in python_sources:
            static_issues.setdefault(name, []).append("no exact digest-matched source copy")
            edges.setdefault(name, set())

    launcher_source = exact_sources.get(LAUNCHER)
    require(launcher_source is not None, "no exact launcher source copy")
    runtime_root, focused_test_roots = infer_launcher_roots(launcher_source)
    require(runtime_root not in static_issues, "runtime root has unresolved static analysis")
    require(not any(root in static_issues for root in focused_test_roots), "test root analysis unresolved")
    runtime_closure = transitive_closure([runtime_root], edges)
    focused_test_closure = transitive_closure(focused_test_roots, edges)

    anonymous_governance_roots = infer_anonymous_governance_roots()
    anonymous_governance_closure = transitive_closure(anonymous_governance_roots, edges)
    expected_anonymous_selection = (
        runtime_closure | anonymous_governance_closure | PROTOCOL_OR_LAUNCHER
    )
    require(
        expected_anonymous_selection == set(anonymous_by_name),
        "anonymous subset does not equal its evidenced runtime/governance/release selection",
    )

    entries: list[dict[str, Any]] = []
    for row in sorted(original_rows, key=lambda item: item["logical_name"]):
        name = row["logical_name"]
        issues = list(static_issues.get(name, []))
        if name not in exact_sources:
            issues.append("no exact digest-matched historical source was found")
        if issues:
            role = "unresolved"
        elif name in runtime_closure:
            role = "runtime_reachable"
        elif name in focused_test_closure:
            role = "test_only"
        elif name in PROTOCOL_OR_LAUNCHER:
            role = "protocol_or_launcher"
        elif name.endswith(".py"):
            role = "other_frozen_dependency"
        else:
            role = "unresolved"
            issues.append("non-Python ledger member has no evidenced launcher/protocol role")
        selected = exact_sources.get(name)
        entries.append(
            {
                "anonymous_subset_member": name in anonymous_by_name,
                "bytes": selected.stat().st_size if selected is not None else None,
                "classification": role,
                "classification_basis": classification_basis(role),
                "exact_source_matches": source_matches[name],
                "ledger_position": row["ledger_position"],
                "local_static_imports": sorted(edges.get(name, set())),
                "logical_name": name,
                "sha256": row["sha256"],
                "unresolved_reasons": sorted(set(issues)),
            }
        )

    classification_counts = Counter(entry["classification"] for entry in entries)
    anonymous_classification_counts = Counter(
        entry["classification"] for entry in entries if entry["anonymous_subset_member"]
    )
    vocabulary = (
        "runtime_reachable",
        "test_only",
        "protocol_or_launcher",
        "other_frozen_dependency",
        "unresolved",
    )
    unresolved_entries = sorted(
        entry["logical_name"] for entry in entries if entry["classification"] == "unresolved"
    )

    runtime_trace = run_import_only_trace(runtime_closure, module_to_file)
    live_manifest_map = {row["relative_path"]: row["sha256"] for row in live_manifest_rows}
    round02_manifest_map = {
        row["relative_path"]: row["sha256"] for row in round02_manifest_rows
    }
    changed_since_round02 = sorted(
        name
        for name in set(live_manifest_map) | set(round02_manifest_map)
        if live_manifest_map.get(name) != round02_manifest_map.get(name)
    )
    changed_code_since_round02 = sorted(
        name for name in changed_since_round02 if name.startswith("code/")
    )
    require(not changed_code_since_round02, "anonymous code subset drifted after round 2")

    report_status = "passed" if not unresolved_entries else "incomplete_unresolved_entries"
    if runtime_trace["status"] != "completed" and report_status == "passed":
        report_status = "passed_static_only"

    report: dict[str, Any] = {
        "audit_id": "RP04-round-02-execution-closure-reconciliation",
        "schema_version": "1.0.0",
        "status": report_status,
        "scope": {
            "claim": (
                "The 18-file anonymous source ledger is a strict selected subset of a "
                "135-entry original ledger whose 135 digests are all unique; it is not "
                "the result of duplicate-digest removal."
            ),
            "classification_vocabulary": [
                "runtime_reachable",
                "test_only",
                "protocol_or_launcher",
                "other_frozen_dependency",
                "unresolved",
            ],
            "path_policy": (
                "Only normalized logical names and relative evidence identifiers are emitted; "
                "original paths and snapshot source_path fields are excluded."
            ),
        },
        "input_bindings": {
            "original_execution_ledger": {
                "identifier": (
                    "results/gpu-qwen35-vllm-paged-multifork-resident-20260814a/"
                    "code.sha256"
                ),
                "sha256": sha256_file(ORIGINAL_LEDGER),
                "bytes": ORIGINAL_LEDGER.stat().st_size,
            },
            "anonymous_source_ledger": {
                "identifier": "supplement_anonymous/code/EXECUTED_SOURCE_SHA256",
                "sha256": sha256_file(ANONYMOUS_LEDGER),
                "bytes": ANONYMOUS_LEDGER.stat().st_size,
            },
            "anonymous_manifest": {
                "identifier": "supplement_anonymous/ANONYMOUS_MANIFEST.sha256",
                "sha256": sha256_file(ANONYMOUS_MANIFEST),
                "entry_count": len(live_manifest_rows),
            },
            "round02_snapshot_manifest": {
                "identifier": "review/round_02/submission/MANIFEST.json",
                "sha256": sha256_file(ROUND02_MANIFEST),
                "snapshot_sha256": round02_payload["snapshot_sha256"],
                "entry_count": len(round02_payload["files"]),
            },
        },
        "original_ledger_summary": {
            "entry_count": len(original_rows),
            "unique_logical_name_count": len(set(logical_names)),
            "unique_digest_count": len(set(digests)),
            "all_digests_unique": len(set(digests)) == len(original_rows),
            "path_independent_ledger_sha256": canonical_ledger_sha256(
                (row["logical_name"], row["sha256"]) for row in original_rows
            ),
            "exact_source_matched_count": sum(bool(source_matches[name]) for name in logical_names),
            "exact_source_unmatched_count": sum(not source_matches[name] for name in logical_names),
            "exact_source_unmatched_entries": sorted(
                name for name in logical_names if not source_matches[name]
            ),
            "repository_worktree_digest_drift_count": len(worktree_drift),
            "repository_worktree_digest_drift_entries": worktree_drift,
        },
        "anonymous_subset_reconciliation": {
            "entry_count": len(anonymous_rows),
            "unique_logical_name_count": len(set(anonymous_names)),
            "unique_digest_count": len(set(anonymous_digests)),
            "all_digests_unique": len(set(anonymous_digests)) == len(anonymous_rows),
            "matched_to_original_count": len(anonymous_rows) - len(anonymous_unmatched),
            "matched_to_original_entries": sorted(set(anonymous_by_name) - set(anonymous_unmatched)),
            "unmatched_to_original_count": len(anonymous_unmatched),
            "unmatched_to_original_entries": anonymous_unmatched,
            "omitted_original_entry_count": len(original_rows) - len(anonymous_rows),
            "is_strict_subset": set(anonymous_by_name) < set(original_by_name),
            "is_digest_deduplication": False,
            "selection_identity": {
                "runtime_static_closure_count": len(runtime_closure),
                "anonymous_governance_test_root_count": len(anonymous_governance_roots),
                "protocol_or_launcher_count": len(PROTOCOL_OR_LAUNCHER),
                "union_count": len(expected_anonymous_selection),
                "union_equals_anonymous_subset": True,
            },
            "explanation": (
                "All 135 original digests are already distinct. The 18 released rows are "
                "selected runtime imports, the anonymous package's three governance-test "
                "roots, and two release/protocol files."
            ),
            "classification_counts": {
                role: anonymous_classification_counts.get(role, 0) for role in vocabulary
            },
            "original_focused_test_entries_omitted_from_anonymous_subset": sorted(
                (focused_test_closure - runtime_closure) - set(anonymous_by_name)
            ),
        },
        "classification_summary": {
            "counts": {role: classification_counts.get(role, 0) for role in vocabulary},
            "classified_entry_count": len(entries) - len(unresolved_entries),
            "unresolved_entry_count": len(unresolved_entries),
            "unresolved_entries": unresolved_entries,
            "other_frozen_dependency_semantics": (
                "This category means the file was frozen by the launcher's all-Python-file "
                "ledger but was not reachable from the known target runtime or focused-test "
                "roots. It does not claim the target runtime imported the file."
            ),
        },
        "static_import_analysis": {
            "method": (
                "Python AST import edges over exact digest-matched bytes, including literal "
                "dynamic imports; non-literal dynamic imports and dynamic code are unresolved."
            ),
            "runtime_root": runtime_root,
            "runtime_closure_count": len(runtime_closure),
            "runtime_closure": sorted(runtime_closure),
            "focused_test_roots": focused_test_roots,
            "focused_test_closure_count": len(focused_test_closure),
            "focused_test_closure": sorted(focused_test_closure),
            "test_only_count": len(focused_test_closure - runtime_closure),
            "test_only": sorted(focused_test_closure - runtime_closure),
            "anonymous_governance_roots": anonymous_governance_roots,
            "anonymous_governance_closure_count": len(anonymous_governance_closure),
            "analysis_issue_count": len(static_issues),
            "analysis_issues": {key: value for key, value in sorted(static_issues.items())},
        },
        "runtime_module_trace": runtime_trace,
        "round02_snapshot_binding": {
            "snapshot_identity_verified": True,
            "snapshot_files_verified": len(round02_payload["files"]),
            "round02_anonymous_manifest_sha256": sha256_file(ROUND02_ANONYMOUS_MANIFEST),
            "round02_anonymous_manifest_entry_count": len(round02_manifest_rows),
            "live_anonymous_manifest_sha256": sha256_file(ANONYMOUS_MANIFEST),
            "live_anonymous_manifest_entry_count": len(live_manifest_rows),
            "code_subset_unchanged_since_round02": True,
            "non_code_derivative_changes_since_round02": changed_since_round02,
        },
        "reconciliation_assertions": {
            "original_ledger_has_135_rows": len(original_rows) == 135,
            "original_ledger_has_135_unique_names": len(set(logical_names)) == 135,
            "original_ledger_has_135_unique_digests": len(set(digests)) == 135,
            "all_original_rows_have_exact_source_bytes": all(source_matches[name] for name in logical_names),
            "anonymous_ledger_has_18_rows": len(anonymous_rows) == 18,
            "anonymous_ledger_has_18_unique_digests": len(set(anonymous_digests)) == 18,
            "anonymous_is_strict_subset": set(anonymous_by_name) < set(original_by_name),
            "anonymous_is_not_digest_deduplication": True,
            "round02_code_subset_matches_live_subset": snapshot_anon_by_name == anonymous_by_name,
            "runtime_trace_matches_static_closure": (
                runtime_trace.get("equals_static_runtime_closure") is True
                if runtime_trace["status"] == "completed"
                else None
            ),
        },
        "path_independent_ledger": entries,
    }
    assert_privacy_safe(report)
    write_json_atomic(OUTPUT, report)
    return 0 if not unresolved_entries else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, json.JSONDecodeError, ReconciliationError) as error:
        print(
            f"provenance reconciliation failed closed ({type(error).__name__})",
            file=sys.stderr,
        )
        raise SystemExit(1)
