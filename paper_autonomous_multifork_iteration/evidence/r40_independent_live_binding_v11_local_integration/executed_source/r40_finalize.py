from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
def req(x,m):
    if not x:raise RuntimeError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seal_ok(value):
    observed=value.get("payload_sha256");copy=dict(value);copy["payload_sha256"]=None;return isinstance(observed,str) and hashlib.sha256(json.dumps(copy,sort_keys=True,separators=(",", ":")).encode()).hexdigest()==observed
def finalize(root,pre,ranks=8):
    req(sorted(p.name for p in root.glob("rank-*") if p.is_dir())==[f"rank-{i}" for i in range(ranks)],"rank directory universe drift")
    req(not any(p.name.startswith(".") or ".tmp" in p.name or p.suffix==".rejected" for p in root.rglob("*") if p.is_file()),"orphan artifact present")
    results=[];primary=[];bindings=None
    for rank in range(ranks):
        binding=root/f"rank-{rank}/raw/real-binding.json";absence=root/f"rank-{rank}/raw/global-absence.json"
        req(binding.is_file() and absence.is_file(),"rank artifact missing")
        req(sorted(p.name for p in binding.parent.iterdir())==["global-absence.json","real-binding.json"],"rank raw file universe drift")
        value=json.loads(binding.read_text());zero=json.loads(absence.read_text());req(seal_ok(value) and seal_ok(zero),"rank/absence seal drift")
        rank_fields={"schema_version","experiment_id","rank","selected_cell","phase_order","phase_receipts","source_reference_coordinate_count","actual_selected_rows_verified","actual_storage_rows_verified","count_vector","real_builder_verified","actual_phase_serializer_verified","off_path_candidate_detector_used","producer_coverage","primary_memory_hook_events","global_hook_counters","execution_bindings","lineage_summary","lineage_summary_sha256","formal_gpu_execution","payload_sha256"};req(set(value)==rank_fields,"rank artifact exact schema drift")
        req(set(zero)=={"schema_version","rank","selected_builds","selected_phases","primary_memory_calls_observed","primary_memory_hook_events","expected_primary_memory_calls","primary_call_coverage_proof","primary_absence_proof","payload_sha256"} and zero["schema_version"]=="forkaudit-r40-v11-global-absence-v1","absence exact schema drift")
        req(value.get("schema_version")=="forkaudit-r40-v11-real-binding-rank-v1" and value.get("experiment_id")==pre["experiment_id"],"rank schema/experiment drift")
        req(value["rank"]==rank and value["selected_cell"]==pre["selected_cell"],"rank/cell drift");req(value["phase_order"]==["setup_pre_transition","post_transition","post_generation"],"phase drift")
        req(value["source_reference_coordinate_count"]==pre["acceptance"]["source_reference_coordinates"] and value["actual_selected_rows_verified"]==18 and value["actual_storage_rows_verified"]==1620,"count drift")
        req(value["lineage_summary"].get("policy")=="materialized" and value["lineage_summary"].get("request_count")==8 and value["lineage_summary"].get("source_coordinate_count")==pre["acceptance"]["all_source_coordinates"],"lineage geometry drift")
        req(value["lineage_summary"]["captured_lineage_edges"]==pre["acceptance"]["materialized_direct_clone_edges_per_rank"] and value["lineage_summary"]["all_edges_direct_aten_clone"] is True,"lineage drift")
        req(value["lineage_summary_sha256"]==hashlib.sha256(json.dumps(value["lineage_summary"],sort_keys=True,separators=(",", ":")).encode()).hexdigest(),"lineage summary hash drift")
        req(set(value["execution_bindings"])==set(pre["required_execution_binding_fields"]),"execution binding fields")
        if bindings is None:bindings=value["execution_bindings"]
        req(value["execution_bindings"]==bindings,"cross-rank execution binding drift")
        req(len(value["phase_receipts"])==3 and all(set(r)=={"phase","selected_rows_verified","full_live_rows_verified","lifecycle_edges_verified","actual_storage_rows_verified","actual_serializer_compared","artifact_relative_path","artifact_sha256","artifact_bytes","gdn_sha256"} for r in value["phase_receipts"]) and [r["lifecycle_edges_verified"] for r in value["phase_receipts"]]==[0,60,420] and [r["selected_rows_verified"] for r in value["phase_receipts"]]==pre["acceptance"]["selected_rows_by_phase"] and [r["actual_storage_rows_verified"] for r in value["phase_receipts"]]==pre["acceptance"]["storage_rows_by_phase"],"row vectors/schema drift")
        req(all(isinstance(r.get("artifact_relative_path"),str) and isinstance(r.get("artifact_sha256"),str) and len(r["artifact_sha256"])==64 and isinstance(r.get("gdn_sha256"),str) and len(r["gdn_sha256"])==64 and type(r.get("artifact_bytes")) is int and r["artifact_bytes"]>0 for r in value["phase_receipts"]),"phase artifact receipt drift")
        for receipt in value["phase_receipts"]:
            artifact=(root.parent/receipt["artifact_relative_path"]).resolve();req(root.parent.resolve() in artifact.parents and artifact.is_file(),"phase artifact missing on finalizer reread");payload=artifact.read_bytes();req(len(payload)==receipt["artifact_bytes"] and hashlib.sha256(payload).hexdigest()==receipt["artifact_sha256"],"phase artifact finalizer bytes/hash drift");disk=json.loads(payload)["gdn_phase_witness"];req(hashlib.sha256(json.dumps(disk,sort_keys=True,separators=(",", ":")).encode()).hexdigest()==receipt["gdn_sha256"],"phase artifact finalizer GDN hash drift")
        expected_calls=pre["acceptance"]["primary_memory_calls_per_rank"]
        req(zero["rank"]==rank and zero.get("expected_primary_memory_calls")==expected_calls and zero.get("primary_memory_calls_observed")==expected_calls and zero.get("primary_call_coverage_proof") is True and zero["primary_absence_proof"] is True and zero["primary_memory_hook_events"]==0,"primary absence proof drift")
        primary.append(zero["primary_memory_hook_events"]);results.append({"rank":rank,"binding_sha256":sha(binding),"absence_sha256":sha(absence),"selected_rows":value["actual_selected_rows_verified"],"storage_rows":value["actual_storage_rows_verified"],"clone_edges":value["lineage_summary"]["captured_lineage_edges"],"phase_artifacts":len(value["phase_receipts"]),"primary_calls":zero["primary_memory_calls_observed"]})
    out={"schema_version":"forkaudit-r40-v11-clean-aggregate-v1","rank_count":len(results),"rank_results":results,"total_selected_rows":sum(r["selected_rows"] for r in results),"total_storage_rows":sum(r["storage_rows"] for r in results),"total_clone_edges":sum(r["clone_edges"] for r in results),"total_phase_artifacts":sum(r["phase_artifacts"] for r in results),"total_primary_calls_observed":sum(r["primary_calls"] for r in results),"primary_events_by_rank":primary,"global_primary_memory_hook_events":sum(primary),"execution_bindings":bindings,"requires_cuda_smoke_before_science":True,"formal_gpu_execution":"clean-only-not-fault-campaign"};req(out["global_primary_memory_hook_events"]==0,"global primary events");return out
def main():
    p=argparse.ArgumentParser();p.add_argument("--capture-root",type=Path,required=True);p.add_argument("--preregistration",type=Path,required=True);p.add_argument("--expected-prereg-sha256",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();req(sha(a.preregistration)==a.expected_prereg_sha256,"prereg hash");out=finalize(a.capture_root,json.loads(a.preregistration.read_text()));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,sort_keys=True,separators=(",", ":"))+"\n",encoding="utf-8",errors="strict") if not a.output.exists() else (_ for _ in ()).throw(FileExistsError("aggregate overwrite"));return 0
if __name__=="__main__":raise SystemExit(main())
