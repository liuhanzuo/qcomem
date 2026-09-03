from __future__ import annotations
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"executed_source"))
from r40_finalize import finalize  # noqa:E402
def seal(v):v=dict(v);v["payload_sha256"]=None;v["payload_sha256"]=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",", ":")).encode()).hexdigest();return v
class Finalizer(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);artifact_tmp=tempfile.TemporaryDirectory(dir=root.parent);self.addCleanup(artifact_tmp.cleanup);artifact_root=Path(artifact_tmp.name);pre=json.loads((ROOT/"preregistration.json").read_text());bindings={k:"a"*64 for k in pre["required_execution_binding_fields"]}
        summary={"policy":"materialized","request_count":8,"source_coordinate_count":60,"captured_lineage_edges":480,"all_edges_direct_aten_clone":True,"source_values_rechecked_unchanged":True}
        for rank in range(8):
            phase_receipts=[]
            for phase in range(3):
                phase_name=["setup_pre_transition","post_transition","post_generation"][phase];artifact=artifact_root/f"rank-{rank}-phase-{phase_name}.json";gdn={"phase":phase_name};binding={"rank":rank,"resident_count":8,"kv_policy":pre["selected_cell"]["kv_policy"],"gdn_base_policy":pre["selected_cell"]["gdn_base_policy"],"phase":phase_name,"cell_id":f"rank-{rank}-N-8-fixture-ownership-witness"};payload=(json.dumps({"binding":binding,"gdn_phase_witness":gdn},sort_keys=True,separators=(",", ":"))+"\n").encode();artifact.write_bytes(payload)
                phase_receipts.append({"phase":phase_name,"selected_rows_verified":6,"full_live_rows_verified":540,"generation_calls_verified":[0,1,64][phase],"request_rebind_counts":[[0]*8,[1,0,0,0,0,0,0,0],[8]*8][phase],"per_call_isolation_verified":True,"actual_serializer_compared":True,"actual_storage_rows_verified":540,"artifact_relative_path":artifact.relative_to(root.parent).as_posix(),"artifact_sha256":hashlib.sha256(payload).hexdigest(),"artifact_bytes":len(payload),"gdn_sha256":hashlib.sha256(json.dumps(gdn,sort_keys=True,separators=(",", ":")).encode()).hexdigest()})
            value=seal({"schema_version":"forkaudit-r40-v13-real-binding-rank-v1","experiment_id":pre["experiment_id"],"rank":rank,"selected_cell":pre["selected_cell"],"phase_order":["setup_pre_transition","post_transition","post_generation"],"source_reference_coordinate_count":60,"actual_selected_rows_verified":18,"actual_storage_rows_verified":1620,"count_vector":{"source_reference_coordinates":60,"selected_rows_by_phase":[6,6,6],"storage_rows_by_phase":[540,540,540],"full_live_rows_by_phase":[540,540,540],"generation_calls_by_phase":[0,1,64],"request_rebind_counts_by_phase":[[0]*8,[1,0,0,0,0,0,0,0],[8]*8],"primary_memory_hook_events":0},"real_builder_verified":True,"actual_phase_serializer_verified":True,"off_path_candidate_detector_used":False,"producer_coverage":{"prebuild_reference_frozen":True,"real_group_observed":True,"actual_serializer_rows_observed":True,"persistent_rechecked_each_phase":True,"all_storage_rows_normalized_against_live_keys":True},"primary_memory_hook_events":0,"global_hook_counters":{"selected_builds":1,"selected_phases":3,"primary_memory_calls_observed":7,"primary_memory_hook_events":0},"formal_gpu_execution":"fixture","lineage_summary":summary,"lineage_summary_sha256":hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(",", ":")).encode()).hexdigest(),"execution_bindings":bindings,"phase_receipts":phase_receipts})
            zero=seal({"schema_version":"forkaudit-r40-v13-global-absence-v1","rank":rank,"selected_builds":1,"selected_phases":3,"expected_primary_memory_calls":12,"primary_memory_calls_observed":12,"primary_call_coverage_proof":True,"primary_absence_proof":True,"primary_memory_hook_events":0})
            path=root/f"rank-{rank}/raw";path.mkdir(parents=True);(path/"real-binding.json").write_text(json.dumps(value));(path/"global-absence.json").write_text(json.dumps(zero))
        return root,pre
    def test_exact_aggregate_derived(self):
        root,pre=self.fixture();out=finalize(root,pre);self.assertEqual((out["total_selected_rows"],out["total_storage_rows"],out["total_clone_edges"],out["total_phase_artifacts"],out["total_primary_calls_observed"]),(144,12960,3840,24,96));self.assertEqual(out["global_primary_memory_hook_events"],0)
    def test_orphan_and_nonzero_primary_rejected(self):
        root,pre=self.fixture();(root/"orphan.tmp").write_text("x")
        with self.assertRaisesRegex(RuntimeError,"orphan|file closure"):finalize(root,pre)
        (root/"orphan.tmp").unlink();path=root/"rank-0/raw/global-absence.json";value=json.loads(path.read_text());value["primary_memory_hook_events"]=1;path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError,"absence proof"):finalize(root,pre)
    def test_extra_rank_and_extra_raw_file_rejected(self):
        root,pre=self.fixture();(root/"rank-8").mkdir()
        with self.assertRaisesRegex(RuntimeError,"rank directory"):finalize(root,pre)
        (root/"rank-8").rmdir();(root/"rank-0/raw/alien.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError,"file closure"):finalize(root,pre)
    def test_resealed_authorizing_flag_and_bool_counter_rejected(self):
        root,pre=self.fixture();path=root/"rank-0/raw/real-binding.json";value=json.loads(path.read_text());value["real_builder_verified"]=1;path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError,"authorizing flag"):finalize(root,pre)
        root,pre=self.fixture();path=root/"rank-0/raw/global-absence.json";value=json.loads(path.read_text());value["selected_builds"]=True;path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError,"absence|type|schema"):finalize(root,pre)
    def test_bool_rank_uppercase_binding_and_cross_rank_artifact_reuse_rejected(self):
        root,pre=self.fixture();path=root/"rank-1/raw/real-binding.json";value=json.loads(path.read_text());value["rank"]=True;path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError,"rank/cell"):finalize(root,pre)
        root,pre=self.fixture();path=root/"rank-0/raw/real-binding.json";value=json.loads(path.read_text());key=next(iter(value["execution_bindings"]));value["execution_bindings"][key]="A"*64;path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError,"lowercase SHA"):finalize(root,pre)
        root,pre=self.fixture();p0=root/"rank-0/raw/real-binding.json";p1=root/"rank-1/raw/real-binding.json";v0=json.loads(p0.read_text());v1=json.loads(p1.read_text());v1["phase_receipts"][0]=dict(v0["phase_receipts"][0]);p1.write_text(json.dumps(seal(v1)))
        with self.assertRaisesRegex(RuntimeError,"reused|rank-cell-phase"):finalize(root,pre)
    def test_benign_extra_capture_file_rejected(self):
        root,pre=self.fixture();(root/"README.txt").write_text("extra")
        with self.assertRaisesRegex(RuntimeError,"file closure"):finalize(root,pre)
if __name__=="__main__":unittest.main()
