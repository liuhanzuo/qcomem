from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,sys,torch
from pathlib import Path
from types import SimpleNamespace
from r40_passive_clone_lineage import PassiveCloneLineageMode,PersistentSourceRegistry
def req(x,m):
    if not x:raise RuntimeError(m)
def owner(device):
    return SimpleNamespace(layers=[SimpleNamespace(conv_states={0:torch.arange(8,device=device,dtype=torch.bfloat16)[::2]},recurrent_states={0:torch.arange(8,device=device,dtype=torch.bfloat16)[1::2]}) for _ in range(30)])
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--runner",type=Path,required=True);a=p.parse_args();req(not a.output.exists(),"smoke overwrite");req(hashlib.sha256(a.runner.read_bytes()).hexdigest()=="9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775","real runner source hash drift");spec=importlib.util.spec_from_file_location("r40_v12_smoke_runner",a.runner);runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner);req(all(callable(getattr(runner,name,None)) for name in ("_run_clean_memory_cell","_run_formal_factorial_cells","_run_ownership_witness_cell","build_resident_request_group")),"real runner interface drift");req(len(runner.FORMAL_RESIDENT_COUNTS)*len(runner.ARM_IDS)==12,"runner protocol primary-call derivation drift");req(torch.__version__=="2.11.0+cu129","formal torch version drift");req(torch.cuda.is_available(),"CUDA unavailable")
    from qcomem_vllm_paged_multifork_resident import _prepare_request_gdn_base,GDN_MATERIALIZE_REQUEST_BASE
    persistent=owner("cuda");coords=[(i,f,0) for i in range(30) for f in ("conv","recurrent")];registry=PersistentSourceRegistry(persistent,coords);requests=[];mode=PassiveCloneLineageMode(registry);plan=SimpleNamespace(linear_layer_indices=tuple(range(30)))
    with mode:
        for _ in range(8):
            request=SimpleNamespace(layers=[SimpleNamespace(conv_states={0:persistent.layers[i].conv_states[0]},recurrent_states={0:persistent.layers[i].recurrent_states[0]}) for i in range(30)])
            _prepare_request_gdn_base(persistent,request,plan,policy=GDN_MATERIALIZE_REQUEST_BASE);requests.append(request)
    result=mode.verify_materialized(requests,coords,require_direct_clone=True);req(result["captured_lineage_edges"]==480,"CUDA clone cardinality")
    torch.cuda.synchronize();probe=torch.empty(1024,device="cuda");pointer=probe.untyped_storage().data_ptr();del probe;torch.cuda.empty_cache();torch.cuda.synchronize();replacement=torch.empty(1024,device="cuda");allocator_reuse_observed=replacement.untyped_storage().data_ptr()==pointer
    receipt={"schema_version":"forkaudit-r40-v12-cuda-smoke-v1","passed":True,"torch_version":torch.__version__,"cuda_bf16":True,"noncontiguous_sources":True,"actual_frozen_helper":True,"clone_edges":480,"storage_interval_checked":True,"allocator_reuse_probe_completed":True,"allocator_reuse_observed":allocator_reuse_observed,"cuda_synchronized":True,"requires_cuda_smoke_before_science":True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,sort_keys=True,separators=(",", ":"))+"\n",encoding="utf-8");return 0
if __name__=="__main__":raise SystemExit(main())
