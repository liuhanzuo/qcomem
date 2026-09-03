from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "r40_independent_live_binding_v30_borrowed_transition_fix"
RELATIVES = ("preregistration.json","absorbed-lineage.json","executed_source/r40_compact_rebind_fix.py","executed_source/r40_no_bytecode_python","executed_source/r40_cuda_smoke.py","executed_source/r40_finalize.py","executed_source/r40_formal_preflight.py","executed_source/r40_passive_clone_lineage.py","executed_source/r40_rank_entrypoint.py","executed_source/r40_real_binding.py","executed_source/r40_real_binding_hook.py","executed_source/r40_tree_closure.py")
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for relative in RELATIVES:
    old,new=BASE/relative,ROOT/relative
    identical=old.read_bytes()==new.read_bytes()
    rows.append({"path":relative,"classification":"byte-identical" if identical else "v31-identity-or-canonical-descriptor-fix","byte_identical":identical,"v30_sha256":sha(old),"v30_size":old.stat().st_size,"v31_sha256":sha(new),"v31_size":new.stat().st_size})
manifest="".join(f"{sha(ROOT/relative)}  {relative}\n" for relative in RELATIVES)
(ROOT/"v31-current-payload.sha256").write_text(manifest,encoding="ascii")
from build_formal_launcher import read_regular_snapshot,transform
v6=ROOT.parent/"r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh"
launcher=hashlib.sha256(transform(read_regular_snapshot(v6).decode()).encode()).hexdigest()
value={"schema_version":"forkaudit-r40-v30-v31-controlled-diff-v1","status":"HOLD_PENDING_FRESH_AUDIT_AND_H20","base_package":BASE.name,"current_package":ROOT.name,"base_payload_manifest_sha256":sha(ROOT/"v30-current-payload.sha256"),"current_payload_manifest_sha256":sha(ROOT/"v31-current-payload.sha256"),"base_payload_file_count":len(rows),"current_payload_file_count":len(rows),"byte_identical_file_count":sum(r["byte_identical"] for r in rows),"controlled_change_file_count":sum(not r["byte_identical"] for r in rows),"immutable_external_runner_sha256":"9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775","immutable_external_builder_sha256":"546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e","v30_generated_launcher_sha256":"ef43b7c1fc80498c7d0c5e36637ced135dda2d9c8281414d4999e57fafefba91","v31_generated_launcher_sha256":launcher,"rows":rows,"payload_sha256":None}
value["payload_sha256"]=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
(ROOT/"v30-v31-controlled-diff.json").write_text(json.dumps(value,indent=2)+"\n")
print(launcher)
