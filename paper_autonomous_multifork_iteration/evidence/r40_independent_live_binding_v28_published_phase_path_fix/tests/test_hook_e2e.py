from __future__ import annotations
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"executed_source"))
from r40_real_binding import digest,storage_key  # noqa:E402
from r40_finalize import finalize  # noqa:E402
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
        artifact_root=root/"primary/raw/.forkaudit-rank-0-fixture";artifact_root.mkdir(parents=True,exist_ok=False)
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
            calls["phase"]+=1;gdn=phase_payload(persistent,group,phase,completed_request_indices);wrapper={"schema_version":"fixture","binding":{"rank":0,"run_id":"fixture","cell_id":"rank-0-N-8-fixture-ownership-witness","resident_count":8,"kv_policy":group.policy,"gdn_base_policy":group.audit["gdn_base_policy"],"gdn_policy":group.audit["gdn_base_policy"],"phase":phase},"gdn_phase_witness":gdn,"kv_ownership_witness":{}};data=(json.dumps(wrapper,sort_keys=True,separators=(",", ":"))+"\n").encode();path=artifact_root/path_prefix/f"phase-{phase}.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);ref={"relative_path":path.relative_to(artifact_root).as_posix(),"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
            if tamper:ref["sha256"]="0"*64
            return ref,gdn
        def witness(**kwargs):
            override=build_override or {};group=runner.build_resident_request_group(persistent,plan,resident_count=override.get("resident_count",kwargs["resident_count"]),policy=override.get("policy",kwargs["kv_policy"]),gdn_base_policy=override.get("gdn_base_policy",kwargs["gdn_base_policy"]))
            common=dict(artifact_root=artifact_root,path_prefix="rank-0/N-8/arm-fixture/witness",persistent=persistent,group=group)
            runner._write_witness_phase(**common,phase="setup_pre_transition",completed_request_indices=[])
            def after(round_index,request_index):
                if round_index==0 and request_index==0:runner._write_witness_phase(**common,phase="post_transition",completed_request_indices=[0])
            runner._gpu_round_robin_generate(group=group,after_step=after)
            runner._write_witness_phase(**common,phase="post_generation",completed_request_indices=list(range(len(group.requests))));return {"ok":True}
        runner.build_resident_request_group=build;runner._write_witness_phase=write_phase;runner._run_ownership_witness_cell=witness;runner._run_clean_memory_cell=lambda **kwargs:{"memory":True}
        def factorial(**kwargs):
            for _ in range(12):runner._run_clean_memory_cell()
        runner._run_formal_factorial_cells=factorial
        def generation(*,group,after_step,**kwargs):
            for round_index in range(8):
                for request_index,request in enumerate(group.requests):
                    for layer in request.layers:
                        for mapping in (layer.conv_states,layer.recurrent_states):mapping[0]=mapping[0].clone().add_(1)
                    after_step(round_index,request_index)
            return [],None,None
        runner._gpu_round_robin_generate=generation
        return runner,calls
    def bindings(self,pre):return {key:"a"*64 for key in pre["required_execution_binding_fields"]}
    def test_full_install_build_three_phases_artifacts_and_restore(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name);runner,calls=self.runner(root);originals=(runner.build_resident_request_group,runner._write_witness_phase,runner._run_ownership_witness_cell,runner._run_clean_memory_cell,runner._run_formal_factorial_cells);pre=json.loads((ROOT/"preregistration.json").read_text());restore=install_real_binding_hook(runner,pre,capture_root=root/"r40-clean-binding",rank=0,execution_bindings=self.bindings(pre));runner._run_formal_factorial_cells();runner._run_ownership_witness_cell(resident_count=8,kv_policy=pre["selected_cell"]["kv_policy"],gdn_base_policy=pre["selected_cell"]["gdn_base_policy"],arm_id="e2e");restore();self.assertEqual(calls,{"build":1,"phase":3});self.assertEqual((runner.build_resident_request_group,runner._write_witness_phase,runner._run_ownership_witness_cell,runner._run_clean_memory_cell,runner._run_formal_factorial_cells),originals);self.assertTrue((root/"r40-clean-binding/rank-0/raw/real-binding.json").is_file());self.assertEqual(GLOBAL_HOOK_COUNTERS["primary_memory_calls_observed"],12);self.assertEqual(GLOBAL_HOOK_COUNTERS["primary_memory_hook_events"],0)

    def test_temporary_phase_paths_rebind_to_published_tree_and_finalize_after_cleanup(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name).resolve();runner,_=self.runner(root);pre=json.loads((ROOT/"preregistration.json").read_text());capture=root/"r40-clean-binding";restore=install_real_binding_hook(runner,pre,capture_root=capture,rank=0,execution_bindings=self.bindings(pre));runner._run_formal_factorial_cells();GLOBAL_HOOK_COUNTERS["primary_memory_calls_observed"]=7;runner._run_ownership_witness_cell(resident_count=8,kv_policy=pre["selected_cell"]["kv_policy"],gdn_base_policy=pre["selected_cell"]["gdn_base_policy"],arm_id="e2e");restore()
        temporary=next((root/"primary/raw").glob(".forkaudit-rank-0-*"));published=root/"primary/raw/rank-0";temporary.joinpath("rank-0").replace(published);temporary.rmdir();self.assertFalse(temporary.exists())
        binding_path=capture/"rank-0/raw/real-binding.json";binding=json.loads(binding_path.read_text());self.assertTrue(all(row["artifact_relative_path"].startswith("primary/raw/rank-0/") and ".forkaudit-rank" not in row["artifact_relative_path"] and (root/row["artifact_relative_path"]).is_file() for row in binding["phase_receipts"]))
        GLOBAL_HOOK_COUNTERS["primary_memory_calls_observed"]=12;zero={"schema_version":"forkaudit-r40-v16-global-absence-v1","rank":0,**GLOBAL_HOOK_COUNTERS,"expected_primary_memory_calls":12,"primary_call_coverage_proof":True,"primary_absence_proof":True,"payload_sha256":None};zero["payload_sha256"]=hashlib.sha256(json.dumps(zero,sort_keys=True,separators=(",", ":")).encode()).hexdigest();(capture/"rank-0/raw/global-absence.json").write_text(json.dumps(zero,sort_keys=True,separators=(",", ":"))+"\n")
        aggregate=finalize(capture,pre,ranks=1);self.assertEqual(aggregate["total_phase_artifacts"],3)

    def test_primary_call_inside_active_ownership_fails_closed(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name);runner,_=self.runner(root);pre=json.loads((ROOT/"preregistration.json").read_text());original=runner._run_ownership_witness_cell
        def nested(**kwargs):
            runner._run_formal_factorial_cells()
            return original(**kwargs)
        runner._run_ownership_witness_cell=nested;restore=install_real_binding_hook(runner,pre,capture_root=root/"r40-clean-binding",rank=0,execution_bindings=self.bindings(pre))
        try:
            with self.assertRaisesRegex(RuntimeError,"inside active ownership"):runner._run_ownership_witness_cell(resident_count=8,kv_policy=pre["selected_cell"]["kv_policy"],gdn_base_policy=pre["selected_cell"]["gdn_base_policy"],arm_id="e2e")
            self.assertEqual(GLOBAL_HOOK_COUNTERS["primary_memory_hook_events"],1)
        finally:restore()
    def test_artifact_tamper_removes_orphan(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name);runner,_=self.runner(root,tamper=True);pre=json.loads((ROOT/"preregistration.json").read_text());restore=install_real_binding_hook(runner,pre,capture_root=root/"r40-clean-binding",rank=0,execution_bindings=self.bindings(pre));
        try:
            with self.assertRaisesRegex(Exception,"artifact bytes/hash"):runner._run_ownership_witness_cell(resident_count=8,kv_policy=pre["selected_cell"]["kv_policy"],gdn_base_policy=pre["selected_cell"]["gdn_base_policy"],arm_id="e2e")
        finally:restore()
        self.assertFalse(any(root.rglob("phase-*.json")));self.assertFalse((root/"r40-clean-binding/rank-0/raw/real-binding.json").exists())
    def test_active_cell_build_kwargs_and_group_metadata_drift_fail(self):
        pre=json.loads((ROOT/"preregistration.json").read_text())
        for build_override,group_override,pattern in (({"policy":"wrong"},None,"build policy"),({"resident_count":7},None,"build N"),(None,{"resident_count":7},"group N"),(None,{"gdn_base_policy":"wrong"},"group policy")):
            temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name);runner,_=self.runner(root,build_override=build_override,group_override=group_override);restore=install_real_binding_hook(runner,pre,capture_root=root/"r40-clean-binding",rank=0,execution_bindings=self.bindings(pre))
            try:
                with self.assertRaisesRegex(Exception,pattern):runner._run_ownership_witness_cell(resident_count=8,kv_policy=pre["selected_cell"]["kv_policy"],gdn_base_policy=pre["selected_cell"]["gdn_base_policy"],arm_id="e2e")
            finally:restore()
    def test_wrong_cell_is_not_instrumented(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name);runner,_=self.runner(root);pre=json.loads((ROOT/"preregistration.json").read_text());restore=install_real_binding_hook(runner,pre,capture_root=root/"r40-clean-binding",rank=0,execution_bindings=self.bindings(pre))
        try:runner._run_ownership_witness_cell(resident_count=1,kv_policy=pre["selected_cell"]["kv_policy"],gdn_base_policy=pre["selected_cell"]["gdn_base_policy"],arm_id="other")
        finally:restore()
        self.assertFalse((root/"r40-clean-binding/rank-0/raw/real-binding.json").exists())
if __name__=="__main__":unittest.main()
