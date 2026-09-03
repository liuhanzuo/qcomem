#!/usr/bin/env python3
"""Closed structural audit for authority, tests, and every process-launch API."""
from __future__ import annotations

import ast
import json
from pathlib import Path


PROCESS_CALLS = {
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    "concurrent.futures.ProcessPoolExecutor",
    "multiprocessing.Pool",
    "multiprocessing.Process",
    "os.fork",
    "os.forkpty",
    "os.popen",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.startfile",
    "os.system",
    "pty.spawn",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
    "subprocess.run",
}
FORBIDDEN_NAMES = {
    "_AUTHORIZED_PLAN_TOKEN",
    "AuthorizedPlan",
    "Lifecycle",
    "_spawn_worker",
    "spawn_worker",
    "spawn_workers",
    "prepare_authorized_plan",
    "isolated_torch_probe",
}


def _function_ancestors(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    names = []
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(current.name)
    return names


def _qualified(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified(node.value, aliases)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _canonical_process_name(qualified: str | None) -> str | None:
    if qualified is None:
        return None
    for name in PROCESS_CALLS:
        if qualified == name or qualified.endswith(f".{name}"):
            return name
    if qualified.startswith("os."):
        attribute = qualified.rsplit(".", 1)[-1]
        if attribute.startswith("spawn") or attribute.startswith("exec"):
            return f"os.{attribute}"
    return None


def _aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                bound = item.asname or item.name.split(".", 1)[0]
                aliases[bound] = item.name if item.asname else bound
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for item in node.names:
                if item.name != "*":
                    aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None:
                continue
            reference = _qualified(value, aliases)
            canonical = _canonical_process_name(reference)
            if canonical is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and aliases.get(target.id) != canonical:
                    aliases[target.id] = canonical
                    changed = True
    return aliases


def _process_call(node: ast.Call, aliases: dict[str, str]) -> str | None:
    return _canonical_process_name(_qualified(node.func, aliases))


def packaged_test_count(root: Path) -> int:
    count = 0
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                count += 1
    return count


def audit(root: Path) -> dict:
    paths = sorted(root.glob("*.py"))
    trees = {path.name: ast.parse(path.read_text(), str(path)) for path in paths}
    markers = []
    process_sites = []
    forbidden = []
    shell_true = []
    skip_sites = []
    test_sites = []
    for name, tree in trees.items():
        aliases = _aliases(tree)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "FORMAL_AUTHORIZED_LAUNCHER"
                    for target in node.targets
                )
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
            ):
                markers.append(name)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in FORBIDDEN_NAMES:
                    forbidden.append((name, node.name, node.lineno))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                    test_sites.append((name, node.name, node.lineno))
                for decorator in node.decorator_list:
                    rendered = ast.unparse(decorator)
                    if "skip" in rendered or "expectedFailure" in rendered:
                        skip_sites.append((name, node.lineno, rendered))
            if isinstance(node, ast.Call):
                call_name = _process_call(node, aliases)
                if call_name is not None:
                    process_sites.append(
                        {
                            "api": call_name,
                            "file": name,
                            "line": node.lineno,
                            "functions": _function_ancestors(node, parents),
                        }
                    )
                if any(
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                ):
                    shell_true.append((name, node.lineno))
                rendered = ast.unparse(node.func)
                if rendered.endswith(("skip", "skipIf", "skipUnless", "expectedFailure")):
                    skip_sites.append((name, node.lineno, rendered))

    production_sites = [site for site in process_sites if not site["file"].startswith("test")]
    test_process_sites = [site for site in process_sites if site["file"].startswith("test")]
    process_gate = (
        len(production_sites) == 2
        and {site["api"] for site in production_sites}
        == {"subprocess.run", "subprocess.Popen"}
        and all(
            site["file"] == "authorized_launcher_v10.py"
            and "run_authorized_campaign" in site["functions"]
            for site in production_sites
        )
    )
    template = json.loads((root / "operator-binding.template.json").read_text())
    approved = {
        key: value
        for key, value in template["payload"].items()
        if key.startswith("approved_")
    }
    test_count = packaged_test_count(root)
    gates = {
        "all_python_ast": bool(trees),
        "all_process_apis_enumerated": process_gate,
        "all_test_process_apis_test_only": all(
            site["file"] == "test_v10.py" for site in test_process_sites
        ),
        "all_packaged_tests_in_test_module": bool(test_sites)
        and all(site[0] == "test_v10.py" for site in test_sites),
        "all_approved_template_fields_null": len(approved) == 16
        and all(value is None for value in approved.values()),
        "hold_documented": "HOLD_PENDING_FRESH_INDEPENDENT_AUDIT_AND_EXTERNAL_BINDING"
        in (root / "README.md").read_text(),
        "no_forbidden_authority_or_spawn_symbols": not forbidden,
        "no_shell_true": not shell_true,
        "no_skip_or_xfail": not skip_sites,
        "one_formal_launcher": markers == ["authorized_launcher_v10.py"],
        "positive_exact_test_count": test_count > 0,
    }
    return {
        "forbidden": forbidden,
        "gates": gates,
        "packaged_tests": test_count,
        "process_sites": process_sites,
        "schema_version": "forkaudit-v10-static-audit-v1",
        "skip_sites": skip_sites,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "test_sites": test_sites,
    }


if __name__ == "__main__":
    result = audit(Path(__file__).resolve().parent)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    raise SystemExit(result["status"] != "PASS")
