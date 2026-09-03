from __future__ import annotations
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"executed_source"))
from r40_real_binding import digest,storage_key  # noqa:E402
from r40_real_binding_hook import GLOBAL_HOOK_COUNTERS,install_real_binding_hook  # noqa:E402

def owner(offset=0,layers=30):
    return SimpleNamespace(layers=[SimpleNamespace(conv_states={0:torch.tensor([offset+i+1,offset+i+2],dtype=torch.bfloat16)},recurrent_states={0:torch.tensor([offset+i+3,offset+i+4],dtype=torch.bfloat16)}) for i in range(layers)])

def phase_payload(persistent,group,phase,completed):
    storages={};rows=[]
    for kind,index,value in [("persistent",None,persistent)]+[("request",i,r) for i,r in enumerate(group.requests)]:
        for layer in range(30):
            for family in ("conv","recurrent"):
                t=getattr(value.layers[layer],family+"_states")[0]; key=storage_key(t)
                if key not in storages:storages[key]=f"storage-{len(storages):04d}"
                start=t.storage_offset()*t.element_size();end=start+t.numel()*t.element_size()
                rows.append({"owner_kind":kind,"request_index":index,"layer_index":layer,"state_family":family,"state_index":0,"shape":list(t.shape),"stride":list(t.stride()),"storage_offset":t.storage_offset(),"dtype":str(t.dtype),"device":str(t.device),"storage_nbytes":t.untyped_storage().nbytes(),"tensor_nbytes":t.numel()*t.element_size(),"byte_start":start,"byte_end_exclusive":end,"content_sha256":digest(t),"storage_id":storages[key]})
    return {"phase":phase,"storage_witness":{"completed_request_indices":list(completed),"rows":rows}}

class E2E(unittest.TestCase):
    def setUp(self):
        for key in GLOBAL_HOOK_COUNTERS:GLOBAL_HOOK_COUNTERS[key]=0
    def runner(self,root,tamper=False,build_override=None,group_override=None):
        persistent=owner(100);plan=SimpleNamespace(linear_layer_indices=tuple(range(30)));calls={"build":0,"phase":0}
        runner=SimpleNamespace()
        def build(cache,received_plan,*,resident_count,policy,gdn_base_policy):
            calls["build"]+=1;requests=[]
            for _ in range(resident_count):
                request=owner()
                for i in range(30):
                    request.layers[i].conv_states[0]=cache.layers[i].conv_states[0].clone();request.layers[i].recurrent_states[0]=cache.layers[i].recurrent_states[0].clone()
                requests.append(request)
            return SimpleNamespace(requests=tuple(requests),resident_count=(group_override or {}).get("resident_count",resident_count),policy=(group_override or {}).get("policy",policy),audit={"gdn_base_policy":(group_override or {}).get("gdn_base_policy",gdn_base_policy)})
        def write_phase(*,artifact_root,path_prefix,phase,completed_request_indices,persistent,group,**kwargs):
            calls["phase"]+=1;gdn=phase_payload(persistent,group,phase,completed_request_indices);wrapper={"gdn_phase_witness":gdn};data=(json.dumps(wrapper,sort_keys=True,separators=(",", ":"))+"\n").encode();path=artifact_root/path_prefix/f"phase-{phase}.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);ref={"relative_path":path.relative_to(artifact_root).as_posix(),"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
            if tamper:ref["sha256"]="0"*64
            return ref,gdn
        def witness(**kwargs):
            override=build_override or {};group=runner.build_resident_request_group(persistent,plan,resident_count=override.get("resident_count",kwargs["resident_count"]),policy=override.get("policy",kwargs["kv_policy"]),gdn_base_policy=override.get("gdn_base_policy",kwargs["gdn_base_policy"]))
            common=dict(artifact_root=root,path_prefix="raw-phase",persistent=persistent,group=group)
            runner._write_witness_phase(**common,phase="setup_pre_transition",completed_request_indices=[])
            for i in range(30): group.requests[0].layers[i].conv_states[0]=group.requests[0].layers[i].conv_states[0].clone().add_(1);group.requests[0].layers[i].recurrent_states[0]=group.requests[0].layers[i].recurrent_states[0].clone().add_(1)
            runner._write_witness_phase(**common,phase="post_transition",completed_request_indices=[0])
            for r in range(1,len(group.requests)):
                for i in range(30): group.requests[r].layers[i].conv_states[0]=group.requests[r].layers[i].conv_states[0].clone().add_(1);group.requests[r].layers[i].recurrent_states[0]=group.requests[r].layers[i].recurrent_states[0].clone().add_(1)
            runner._write_witness_phase(**common,phase="post_generation",completed_request_indices=list(range(len(group.requests))));return {"ok":True}
        runner.build_resident_request_group=build;runner._write_witness_phase=write_phase;runner._run_ownership_witness_cell=witness;runner._run_memory_cell=lambda **kwargs:{"memory":True}
        return runner,calls
    def bindings(self,pre):return {key:"a"*64 for key in pre["required_execution_binding_fields"]}
    def test_full_install_build_three_phases_artifacts_and_restore(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name);runner,calls=self.runner(root);originals=(runner.build_resident_request_group,runner._write_witness_phase,runner._run_ownership_witness_cell,runner._run_memory_cell);pre=json.loads((ROOT/"preregistration.json").read_text());restore=install_real_binding_hook(runner,pre,capture_root=root,rank=0,execution_bindings=self.bindings(pre));runner._run_memory_cell();runner._run_ownership_witness_cell(resident_count=8,kv_policy=pre["selected_cell"]["kv_policy"],gdn_base_policy=pre["selected_cell"]["gdn_base_policy"],arm_id="e2e");restore();self.assertEqual(calls,{"build":1,"phase":3});self.assertEqual((runner.build_resident_request_group,runner._write_witness_phase,runner._run_ownership_witness_cell,runner._run_memory_cell),originals);self.assertTrue((root/"rank-0/raw/real-binding.json").is_file());self.assertEqual(GLOBAL_HOOK_COUNTERS["primary_memory_calls_observed"],1);self.assertEqual(GLOBAL_HOOK_COUNTERS["primary_memory_hook_events"],0)
    def test_artifact_tamper_removes_orphan(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name);runner,_=self.runner(root,tamper=True);pre=json.loads((ROOT/"preregistration.json").read_text());restore=install_real_binding_hook(runner,pre,capture_root=root,rank=0,execution_bindings=self.bindings(pre));
        try:
            with self.assertRaisesRegex(Exception,"artifact bytes/hash"):runner._run_ownership_witness_cell(resident_count=8,kv_policy=pre["selected_cell"]["kv_policy"],gdn_base_policy=pre["selected_cell"]["gdn_base_policy"],arm_id="e2e")
        finally:restore()
        self.assertFalse(any(root.rglob("phase-*.json")));self.assertFalse((root/"rank-0/raw/real-binding.json").exists())
    def test_active_cell_build_kwargs_and_group_metadata_drift_fail(self):
        pre=json.loads((ROOT/"preregistration.json").read_text())
        for build_override,group_override,pattern in (({"policy":"wrong"},None,"build policy"),({"resident_count":7},None,"build N"),(None,{"resident_count":7},"group N"),(None,{"gdn_base_policy":"wrong"},"group policy")):
            temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name);runner,_=self.runner(root,build_override=build_override,group_override=group_override);restore=install_real_binding_hook(runner,pre,capture_root=root,rank=0,execution_bindings=self.bindings(pre))
            try:
                with self.assertRaisesRegex(Exception,pattern):runner._run_ownership_witness_cell(resident_count=8,kv_policy=pre["selected_cell"]["kv_policy"],gdn_base_policy=pre["selected_cell"]["gdn_base_policy"],arm_id="e2e")
            finally:restore()
    def test_wrong_cell_is_not_instrumented(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name);runner,_=self.runner(root);pre=json.loads((ROOT/"preregistration.json").read_text());restore=install_real_binding_hook(runner,pre,capture_root=root,rank=0,execution_bindings=self.bindings(pre))
        try:runner._run_ownership_witness_cell(resident_count=1,kv_policy=pre["selected_cell"]["kv_policy"],gdn_base_policy=pre["selected_cell"]["gdn_base_policy"],arm_id="other")
        finally:restore()
        self.assertFalse((root/"rank-0/raw/real-binding.json").exists())
if __name__=="__main__":unittest.main()
