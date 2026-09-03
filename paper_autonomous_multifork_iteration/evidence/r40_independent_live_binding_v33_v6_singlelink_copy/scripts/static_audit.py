from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    pre=json.loads((ROOT/"preregistration.json").read_text());hook=(ROOT/"executed_source/r40_real_binding_hook.py").read_text();binding=(ROOT/"executed_source/r40_real_binding.py").read_text();lineage=(ROOT/"executed_source/r40_passive_clone_lineage.py").read_text();finalize=(ROOT/"executed_source/r40_finalize.py").read_text();launcher=(ROOT/"formal/launch_h20.sh").read_text();tree=(ROOT/"executed_source/r40_tree_closure.py").read_text();manifest=json.loads((ROOT/"package-manifest.json").read_text())
    checks={
      "selected_borrowed_cell":pre["selected_cell"]=={"cell_role":"ownership_witness","resident_count":8,"kv_policy":"vllm-q16-shared-document-reuse","gdn_base_policy":"borrow-immutable-base-functional-rebind"},
      "phase_vectors_frozen":pre["acceptance"]["borrowed_exact_alias_rows_by_phase"]==[480,420,0] and pre["acceptance"]["private_request_rows_by_phase"]==[0,60,480] and pre["acceptance"]["functional_rebind_edges_by_phase"]==[0,60,3840],
      "actual_builder_frozen_before_call":"verifier = ActualBindingVerifier(cache" in hook and hook.index("verifier = ActualBindingVerifier(cache")<hook.index("group = original_build(cache, plan, **kwargs)"),
      "borrowed_capability_exact_alias":"borrowed capability destination is not exact persistent alias" in lineage and "verify_borrowed(group.requests" in hook,
      "production_serializer_bound":"result = original_phase(*args, **kwargs)" in hook and "verify_serialized_phase(result[1]" in hook and "actual_storage_rows_verified" in hook,
      "functional_endpoints_recorded":all(token in binding for token in ("object_id","storage_key","descriptor","content_sha256","functional_rebind_receipt")),
      "phase_alias_private_gates":all(token in binding for token in ("incomplete request is not exact read-only persistent alias","completed request aliases persistent base","completed request/peer coordinates alias")),
      "finalizer_expands_all_3840_edges":"validate_functional_ledger" in finalize and "post endpoint reused across ledger" in finalize,
      "finalizer_rank_counter_exact_integer_eight":"primary_memory_calls_observed\": 8" in finalize and "all(type(item) is int for item in counters.values())" in finalize and pre["acceptance"]["rank_artifact_primary_memory_calls_observed"]==8,
      "fixed_v33_paths":"qcomem_r40_v33_v6_singlelink_copy_20260902b" in launcher and "qcomem_r40_v33_v6_singlelink_copy_20260902b" in tree and manifest["result_name"]=="r40-v33-v6-singlelink-copy-20260902b",
      "v32_authority_env":all(token in launcher for token in ("R40_V32_FRESH_AUDIT_APPROVED","R40_V32_APPROVED_SOURCE_LEDGER_SHA256","R40_V32_APPROVED_ARCHIVE_SHA256","R40_V32_OVERLAY_ARCHIVE")),
      "no_v29_control_path_schema":not re.search(r"R40_V29|qcomem_r40_v29|r40-v29|forkaudit-r40-v29",launcher+tree+hook+binding+finalize+json.dumps({key:manifest[key] for key in ("schema_version","package_id","archive_name","stage_name","result_name")},sort_keys=True)),
      "hold_not_evidence":pre["status"]=="HOLD_PENDING_FRESH_AUDIT_AND_H20" and pre["formal_gpu_execution"]=="not-run" and manifest["science_accepted"] is False,
      "source_ledger_present":(ROOT/"source-code.sha256").is_file(),
      "current_payload_present":(ROOT/"v32-current-payload.sha256").is_file() and (ROOT/"v31-v32-controlled-diff.json").is_file(),
    }
    failed=sorted(name for name,value in checks.items() if value is not True)
    output={"schema_version":"forkaudit-r40-v32-borrowed-transition-static-v1","status":"HOLD_PENDING_FRESH_AUDIT_AND_H20","formal_gpu_execution":"not-run","checks":checks,"failed_checks":failed,"all_checks_satisfied":not failed}
    print(json.dumps(output,sort_keys=True,separators=(",", ":")))
    if failed:raise SystemExit("static audit failed: "+",".join(failed))
    return 0
if __name__=="__main__":raise SystemExit(main())
