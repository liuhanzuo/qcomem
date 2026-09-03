#!/usr/bin/env python3
"""Read-only static release audit for the v8 method-freeze source tree."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def audit(root: Path) -> dict:
    python_paths = sorted(root.glob("*.py"))
    parsed = {path.name: ast.parse(path.read_text(), filename=str(path)) for path in python_paths}

    formal_markers = []
    for name, tree in parsed.items():
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
                formal_markers.append(name)

    runtime_tree = parsed["v8_runtime.py"]
    spawn_signatures = []
    public_single_spawn = []
    shell_true_calls = []
    for name, tree in parsed.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if name == "v8_runtime.py" and node.name == "spawn_workers":
                    spawn_signatures.append([argument.arg for argument in node.args.args])
                if name == "v8_runtime.py" and node.name == "spawn_worker":
                    public_single_spawn.append(node.lineno)
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if (
                        keyword.arg == "shell"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        shell_true_calls.append({"file": name, "line": node.lineno})

    template = json.loads((root / "operator-binding.template.json").read_text())
    null_anchors = sorted(
        key for key, value in template["payload"].items() if key.startswith("approved_") and value is None
    )
    expected_null_anchors = sorted(
        key for key in template["payload"] if key.startswith("approved_")
    )
    hold_text = "HOLD_PENDING_FRESH_AUDIT_AND_EXTERNAL_OPERATOR_BINDING"
    gates = {
        "all_python_ast": len(parsed) == len(python_paths) and bool(parsed),
        "binding_template_null_bound": null_anchors == expected_null_anchors,
        "exact_one_formal_launcher": formal_markers == ["authorized_launcher_v8.py"],
        "hold_status_documented": hold_text in (root / "README.md").read_text(),
        "lifecycle_no_public_single_spawn": public_single_spawn == [],
        "lifecycle_spawn_has_no_caller_specs": spawn_signatures == [["self"]],
        "no_shell_true": shell_true_calls == [],
    }
    return {
        "formal_launchers": formal_markers,
        "gates": gates,
        "python_files": [path.name for path in python_paths],
        "schema_version": "forkaudit-v8-static-audit-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def main() -> int:
    result = audit(Path(__file__).resolve().parent)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
