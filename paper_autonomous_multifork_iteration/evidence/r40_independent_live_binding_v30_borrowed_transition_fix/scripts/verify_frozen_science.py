from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,stat
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGE_NAME="r40_independent_live_binding_v30_borrowed_transition_fix"
MANIFEST_NAME="v30-current-payload.sha256"
DIFF_NAME="v29-v30-controlled-diff.json"
RUNNER_SHA="9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775"
BUILDER_SHA="546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e"
RELATIVES=("preregistration.json","absorbed-lineage.json","executed_source/r40_compact_rebind_fix.py","executed_source/r40_no_bytecode_python","executed_source/r40_cuda_smoke.py","executed_source/r40_finalize.py","executed_source/r40_formal_preflight.py","executed_source/r40_passive_clone_lineage.py","executed_source/r40_rank_entrypoint.py","executed_source/r40_real_binding.py","executed_source/r40_real_binding_hook.py","executed_source/r40_tree_closure.py")
def req(x,m):
    if not x:raise RuntimeError(m)
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def stable(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0))
    try:
        before=os.fstat(fd);req(stat.S_ISREG(before.st_mode) and before.st_nlink==1,"frozen payload must be singly linked regular")
        with os.fdopen(fd,"rb",closefd=False) as stream:data=stream.read()
        after=os.fstat(fd);req((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),"frozen payload changed during read");return data
    finally:os.close(fd)
def load_builder(path):
    spec=importlib.util.spec_from_file_location("r40_v30_launcher_builder",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def verify(root=ROOT,repo_root=None):
    root=root.resolve(strict=True);req(root.name==PACKAGE_NAME,"V30 package root drift");repo=(repo_root or root.parents[2]).resolve(strict=True)
    lines=stable(root/MANIFEST_NAME).decode("ascii").splitlines();req(len(lines)==len(RELATIVES),"V30 payload manifest count drift")
    rows=[]
    for relative,line in zip(RELATIVES,lines):
        pieces=line.split("  ",1);req(len(pieces)==2 and pieces[1]==relative and len(pieces[0])==64,"V30 payload manifest row drift");req(sha(root/relative)==pieces[0],f"V30 payload drift: {relative}");rows.append((relative,pieces[0]))
    diff=json.loads(stable(root/DIFF_NAME));observed=diff.get("payload_sha256");candidate=dict(diff);candidate["payload_sha256"]=None;req(observed==sha_bytes(json.dumps(candidate,sort_keys=True,separators=(",", ":")).encode()),"V29-V30 diff seal drift")
    req(diff["schema_version"]=="forkaudit-r40-v29-v30-controlled-diff-v1" and diff["status"]=="HOLD_PENDING_FRESH_AUDIT_AND_H20" and diff["current_package"]==PACKAGE_NAME,"V29-V30 diff identity drift")
    req(diff["current_payload_manifest_sha256"]==sha(root/MANIFEST_NAME) and [row["path"] for row in diff["rows"]]==list(RELATIVES),"V29-V30 diff manifest/path drift")
    for (relative,current_sha),row in zip(rows,diff["rows"]):req(row["v30_sha256"]==current_sha and row["v30_size"]==(root/relative).stat().st_size,"V30 diff current hash/size drift")
    gpu=repo/"paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/gpu";req(sha(gpu/"run_qcomem_qwen35_forkaudit_review_revision.py")==RUNNER_SHA and sha(gpu/"qcomem_vllm_paged_multifork_resident.py")==BUILDER_SHA,"immutable production dependency drift")
    builder=load_builder(root/"scripts/build_formal_launcher.py");v6=repo/"paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh";generated=builder.transform(builder.read_regular_snapshot(v6).decode("utf-8",errors="strict"));generated_sha=sha_bytes(generated.encode());req(generated_sha==diff["v30_generated_launcher_sha256"],"V30 generated launcher hash drift")
    return {"schema_version":"forkaudit-r40-v30-borrowed-transition-payload-verification-v1","status":"HOLD_PENDING_FRESH_AUDIT_AND_H20","science_accepted":False,"current_payload_file_count":len(RELATIVES),"current_payload_manifest_sha256":sha(root/MANIFEST_NAME),"controlled_diff_sha256":sha(root/DIFF_NAME),"v30_generated_launcher_sha256":generated_sha}
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--repo-root",type=Path);a=p.parse_args();print(json.dumps(verify(a.root,a.repo_root),sort_keys=True,separators=(",", ":")));return 0
if __name__=="__main__":raise SystemExit(main())
