#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,os,sys
from pathlib import Path
EXPECTED_BASE="5c5ffdac992b0ee0e4f5f8a42bba0a7ce25749d187f395924f6b34a43a484365"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root,ledger,expected):
    if sha(ledger)!=expected:raise SystemExit("source ledger drift")
    for line in Path(ledger).read_text().splitlines():
        digest,relative=line.split("  ",1)
        if sha(root/relative)!=digest:raise SystemExit(f"source drift {relative}")
def load(path):
    spec=importlib.util.spec_from_file_location("r39_v6_base",path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module
def option(name):
    indices=[i for i,v in enumerate(sys.argv) if v==name]
    if len(indices)!=1:raise SystemExit(f"unique {name} missing")
    return sys.argv[indices[0]+1]
def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream:json.dump(value,stream,sort_keys=True,separators=(",", ":"));stream.write("\n")
def seal(value):
    value=dict(value);value["payload_sha256"]=None;value["payload_sha256"]=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest();return value
def main():
    root=Path(os.environ["R40_ROOT"]).resolve();verify(root,root/"source-code.sha256",os.environ["R40_SOURCE_LEDGER_SHA256"])
    if sha(root/"preregistration.json")!=os.environ["R40_PREREG_SHA256"]:raise SystemExit("prereg drift")
    absorbed=json.loads((root/"absorbed-lineage.json").read_text())
    if absorbed.get("source_ledger_sha256")!="a4d84c29a2da3fa902adaacee4b21ebe9e5b14e8e4f508442cbcecb404c24d17":raise SystemExit("passive lineage authority drift")
    base_path=Path(os.environ["R40_BASE_ENTRYPOINT"])
    if sha(base_path)!=EXPECTED_BASE:raise SystemExit("base entrypoint drift")
    sys.path.insert(0,str(root/"executed_source"));from r40_real_binding_hook import install_real_binding_hook,global_absence_receipt
    from r40_compact_rebind_fix import compact_rebind_receipt,install_compact_rebind_fix
    base=load(base_path);original=base._install_primary_scope_wrappers;pre=json.loads((root/"preregistration.json").read_text());rank=int(option("--rank"))
    def combined(*,runner_module,recorder):
        restore_base=original(runner_module=runner_module,recorder=recorder)
        try:restore_fix,_install=install_compact_rebind_fix(runner_module)
        except BaseException:restore_base();raise
        try:restore_r40=install_real_binding_hook(runner_module,pre,capture_root=Path(os.environ["R40_CAPTURE_ROOT"]),rank=rank,execution_bindings=json.loads(os.environ["R40_BINDINGS_JSON"]))
        except BaseException:restore_fix();restore_base();raise
        def restore():restore_r40();restore_fix();restore_base()
        return restore
    base._install_primary_scope_wrappers=combined;result=base.main()
    producer=compact_rebind_receipt()
    if not (producer["runtime_backbones_hooked"]==1 and producer["groups_wrapped"]>0 and producer["requests_wrapped"]>0 and producer["borrowed_setup_calls_delegated"]>0 and producer["materialized_setup_calls_canonicalized"]>0 and producer["materialized_setup_states_canonicalized"]==60*producer["materialized_requests_returned"] and producer["single_token_cached_calls_observed"]>0 and producer["cached_calls_postprocessed"]>0 and producer["all_single_token_calls_rebound_exactly_30"] is True and producer["all_cached_calls_post_rebound_exactly_30_recurrent_states"] is True and producer["all_cached_calls_postprocessed_exactly_once"] is True and producer["all_materialized_requests_directly_compact_cloned_60_states"] is True and producer["all_request_construction_borrow_steps_delegated"] is True):raise SystemExit("compact rebind producer coverage drift")
    receipt=global_absence_receipt();expected_calls=len(pre["acceptance"]["formal_resident_counts"])*int(pre["acceptance"]["formal_arm_count"]);
    if expected_calls!=int(pre["acceptance"]["primary_memory_calls_per_rank"]):raise SystemExit("primary protocol call derivation drift")
    receipt.update({"schema_version":"forkaudit-r40-v16-global-absence-v1","rank":rank,"expected_primary_memory_calls":expected_calls,"primary_call_coverage_proof":receipt["primary_memory_calls_observed"]==expected_calls,"primary_absence_proof":receipt["primary_memory_calls_observed"]==expected_calls and receipt["primary_memory_hook_events"]==0,"payload_sha256":None})
    write_new(Path(os.environ["R40_CAPTURE_ROOT"])/f"rank-{rank}/raw/global-absence.json",seal(receipt));return int(result or 0)
if __name__=="__main__":raise SystemExit(main())
