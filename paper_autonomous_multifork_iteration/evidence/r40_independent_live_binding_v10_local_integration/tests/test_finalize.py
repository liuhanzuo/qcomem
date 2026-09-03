from __future__ import annotations
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"executed_source"))
from r40_finalize import finalize  # noqa:E402
def seal(v):v=dict(v);v["payload_sha256"]=None;v["payload_sha256"]=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",", ":")).encode()).hexdigest();return v
class Finalizer(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);pre=json.loads((ROOT/"preregistration.json").read_text());bindings={k:"a"*64 for k in pre["required_execution_binding_fields"]}
        summary={"policy":"materialized","request_count":8,"source_coordinate_count":60,"captured_lineage_edges":480,"all_edges_direct_aten_clone":True}
        for rank in range(8):
            phase_receipts=[{"selected_rows_verified":6,"actual_storage_rows_verified":540,"artifact_sha256":"b"*64,"artifact_bytes":100,"gdn_sha256":"c"*64} for _ in range(3)]
            value=seal({"schema_version":"forkaudit-r40-v10-real-binding-rank-v1","experiment_id":pre["experiment_id"],"rank":rank,"selected_cell":pre["selected_cell"],"phase_order":["setup_pre_transition","post_transition","post_generation"],"source_reference_coordinate_count":5,"actual_selected_rows_verified":18,"actual_storage_rows_verified":1620,"lineage_summary":summary,"lineage_summary_sha256":hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(",", ":")).encode()).hexdigest(),"execution_bindings":bindings,"phase_receipts":phase_receipts})
            path=root/f"rank-{rank}/raw";path.mkdir(parents=True);(path/"real-binding.json").write_text(json.dumps(value));(path/"global-absence.json").write_text(json.dumps({"rank":rank,"expected_primary_memory_calls":12,"primary_memory_calls_observed":12,"primary_call_coverage_proof":True,"primary_absence_proof":True,"primary_memory_hook_events":0}))
        return root,pre
    def test_exact_aggregate_derived(self):
        root,pre=self.fixture();out=finalize(root,pre);self.assertEqual((out["total_selected_rows"],out["total_storage_rows"],out["total_clone_edges"],out["total_phase_artifacts"],out["total_primary_calls_observed"]),(144,12960,3840,24,96));self.assertEqual(out["global_primary_memory_hook_events"],0)
    def test_orphan_and_nonzero_primary_rejected(self):
        root,pre=self.fixture();(root/"orphan.tmp").write_text("x")
        with self.assertRaisesRegex(RuntimeError,"orphan"):finalize(root,pre)
        (root/"orphan.tmp").unlink();path=root/"rank-0/raw/global-absence.json";value=json.loads(path.read_text());value["primary_memory_hook_events"]=1;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError,"absence proof"):finalize(root,pre)
if __name__=="__main__":unittest.main()
