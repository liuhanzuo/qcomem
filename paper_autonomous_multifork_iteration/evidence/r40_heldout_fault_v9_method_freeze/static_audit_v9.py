#!/usr/bin/env python3
"""Structural audit: one authority entry and one unreachable process-creation site."""
from __future__ import annotations
import ast, json
from pathlib import Path

def audit(root: Path) -> dict:
    paths=sorted(root.glob("*.py")); trees={p.name:ast.parse(p.read_text(),str(p)) for p in paths}
    markers=[]; popens=[]; forbidden=[]; shell=[]
    for name,tree in trees.items():
        parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="FORMAL_AUTHORIZED_LAUNCHER" for t in node.targets) and isinstance(node.value,ast.Constant) and node.value.value is True: markers.append(name)
            if name in {"authorized_launcher_v9.py","v9_runtime.py"} and isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in {"AuthorizedPlan","Lifecycle","_spawn_worker","spawn_worker","spawn_workers"}: forbidden.append((name,node.name,node.lineno))
            if isinstance(node,ast.Call):
                if name in {"authorized_launcher_v9.py","v9_runtime.py"} and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=="subprocess" and node.func.attr=="Popen":
                    cur=node
                    while cur in parents and not isinstance(cur,(ast.FunctionDef,ast.AsyncFunctionDef)): cur=parents[cur]
                    popens.append((name,getattr(cur,"name",None),node.lineno))
                if any(k.arg=="shell" and isinstance(k.value,ast.Constant) and k.value.value is True for k in node.keywords): shell.append((name,node.lineno))
    runtime=trees["v9_runtime.py"]
    runtime_imports_subprocess=any(isinstance(n,(ast.Import,ast.ImportFrom)) and ((isinstance(n,ast.ImportFrom) and n.module=="subprocess") or (isinstance(n,ast.Import) and any(a.name=="subprocess" for a in n.names))) for n in ast.walk(runtime))
    template=json.loads((root/"operator-binding.template.json").read_text())
    approved={k:v for k,v in template["payload"].items() if k.startswith("approved_")}
    gates={
      "all_python_ast":bool(trees),
      "one_formal_launcher":markers==["authorized_launcher_v9.py"],
      "one_popen_inside_authorized_call":popens==[("authorized_launcher_v9.py","run_authorized_campaign",421)],
      "no_spawn_or_authority_object":forbidden==[],
      "runtime_has_no_subprocess":not runtime_imports_subprocess,
      "all_approved_template_fields_null":len(approved)>=11 and all(v is None for v in approved.values()),
      "no_shell_true":not shell,
      "hold_documented":"HOLD_PENDING_FRESH_AUDIT_AND_EXTERNAL_OPERATOR_BINDING" in (root/"README.md").read_text(),
    }
    return {"schema_version":"forkaudit-v9-static-audit-v1","gates":gates,"popen_sites":popens,"forbidden":forbidden,"status":"PASS" if all(gates.values()) else "FAIL"}

if __name__=="__main__":
    result=audit(Path(__file__).resolve().parent); print(json.dumps(result,sort_keys=True,separators=(",",":"))); raise SystemExit(result["status"]!="PASS")
