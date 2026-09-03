from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    acceptance = json.loads((ROOT / "acceptance.json").read_text())
    hook = (ROOT / "executed_source/r40_real_binding_hook.py").read_text()
    verifier = (ROOT / "executed_source/r40_real_binding.py").read_text()
    tests = (ROOT / "tests/test_real_binding.py").read_text()
    lineage = (ROOT / "executed_source/r40_passive_clone_lineage.py").read_text()
    lineage_tests = (ROOT / "tests/test_strong_lineage_counterexamples.py").read_text()
    absorbed = json.loads((ROOT / "absorbed-lineage.json").read_text())
    for path in list((ROOT / "executed_source").glob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    checks = {
        "runtime_gated_not_run": prereg["formal_gpu_execution"] == "not-run" and acceptance["formal_science_eligible"] is False and acceptance["requires_cuda_smoke_before_science"] is True,
        "formal_launcher_present": (ROOT/"formal/launch_h20.sh").is_file(),
        "reference_before_original_build": hook.index("ActualBindingVerifier(cache") < hook.index("group = original_build"),
        "actual_phase_after_original_return": hook.index("result = original_phase") < hook.index("verify_serialized_phase"),
        "actual_serializer_rows_consumed": 'gdn.get("storage_witness", {}).get("rows")' in verifier,
        "off_path_candidate_absent": "_candidate_items" not in verifier + hook,
        "four_real_fault_tests": sum(name in tests for name in ("coherent_cross_layer_swap", "request_base_alias", "post_rebind_stale", "same_geometry_one_way")) == 4,
        "instrumented_primary_absence": "primary_memory_calls_observed" in hook and "primary_memory_hook_events" in hook,
        "six_selected_coordinates": len(prereg["selected_coordinates"]) == 6,
        "three_audit_regressions_present": all(name in tests for name in ("completed_request_aliasing_incomplete_peer", "forged_normalized_serializer_storage_id", "persistent_mutation_after_prebuild_freeze")),
        "normalized_ids_from_live_keys": "actual serializer normalized storage_id/live-storage mismatch" in verifier,
        "persistent_rechecked": "persistent source content changed after freeze" in verifier,
        "entrypoint_and_finalizer_present": (ROOT/"executed_source/r40_rank_entrypoint.py").is_file() and (ROOT/"executed_source/r40_finalize.py").is_file(),
        "runtime_gate_machine_readable": json.loads((ROOT / "formal-blocker.json").read_text())["status"] == "local-closure-pass-fresh-audit-hold",
        "exact_row_schema_gate": "serializer row exact schema drift" in verifier,
        "row_cardinality_and_unique_gate": "serializer row cardinality drift" in verifier and "serializer duplicate semantic row" in verifier,
        "full_descriptor_interval_gate": all(text in verifier for text in ("serializer stride/offset mismatch","serializer nbytes mismatch","serializer byte interval mismatch")),
        "completed_crosscheck_gate": "completed set independent lifecycle/GDN/call-args mismatch" in verifier,
        "lineage_interface_fail_closed": "opaque mode capability required" in verifier and "lineage capability gate unsatisfied" in hook,
        "exact_clone_edge_binding": all(text in verifier for text in ("aten.clone.default","lineage source object/interval/storage mismatch","lineage destination exact live object mismatch")),
        "torchdispatch_wraps_original_build": "with lineage:" in hook and "group = original_build" in hook and "PassiveCloneLineageMode" in hook,
        "strong_clone_handles": "source: torch.Tensor" in lineage and "destination: torch.Tensor" in lineage,
        "strong_lineage_counterexamples": all(text in lineage_tests for text in ("same_storage_offset_wrong_source","returned_view_object_not_exact","unused_extra_rooted_edge","private_forgery_breaks_seal")),
        "independent_lineage_hash_bound": absorbed["source_ledger_sha256"] == "a4d84c29a2da3fa902adaacee4b21ebe9e5b14e8e4f508442cbcecb404c24d17" and absorbed["prototype_sha256"] == "ca2e1e35a84b47237bb4334d31ecc65a412c95c17f0562d87f588d7e9e3e6f81",
        "sealed_private_dispatch_ledger": "self._events" in lineage and "_verify_event_ledger" in lineage and "hmac.compare_digest" in lineage,
        "opaque_capability_only": "attach_lineage_capability" in verifier and "mapping lineage receipts are forbidden" in verifier and "issue_capability" in lineage,
        "v11_payload_identity": "forkaudit-r40-v11-real-binding-rank-v1" in hook and "forkaudit_r40_v11_real_binding_context" in hook and "r40-v4" not in hook,
        "exact_clean_memory_interface": "_run_clean_memory_cell" in hook and "_run_memory_cell" not in hook and "_run_formal_factorial_cells" in hook,
        "active_primary_fail_closed": "inside active ownership context" in hook and "primary_memory_hook_events" in hook,
        "dispatch_internal_token": "lineage event creation outside __torch_dispatch__ rejected" in lineage and "self.__dispatch_token" in lineage,
        "single_bound_capability": "lineage mode may issue only one capability" in lineage and "lineage capability verifier/group binding drift" in lineage,
        "full_live_lifecycle": "full_live_rows_verified" in verifier and "for request_index,request in enumerate(group.requests)" in verifier,
        "exact_scalar_types": "serializer exact field type drift" in verifier and "completed indices exact type drift" in verifier,
        "finalizer_rereads_actual_artifacts": "phase artifact finalizer bytes/hash drift" in (ROOT/"executed_source/r40_finalize.py").read_text() and "rank directory universe drift" in (ROOT/"executed_source/r40_finalize.py").read_text(),
        "failure_ledger": "forkaudit-r40-v11-launch-failure-v1" in (ROOT/"formal/launch_h20.sh").read_text(),
        "cuda_binds_real_runner": "real runner source hash drift" in (ROOT/"executed_source/r40_cuda_smoke.py").read_text() and "_run_clean_memory_cell" in (ROOT/"executed_source/r40_cuda_smoke.py").read_text(),
        "primary_call_coverage_exact": prereg["acceptance"]["primary_memory_calls_per_rank"] == 12 and "expected_primary_memory_calls" in (ROOT/"executed_source/r40_rank_entrypoint.py").read_text() and "primary_call_coverage_proof" in (ROOT/"executed_source/r40_finalize.py").read_text(),
        "artifact_reread_and_cleanup": "_bind_phase_artifact" in hook and "artifact_path.unlink" in hook,
        "cuda_smoke_before_science": "r40_cuda_smoke.py" in (ROOT/"scripts/build_formal_launcher.py").read_text(),
        "linux_sha_readonly_terminal": "sha256sum -c" in (ROOT/"scripts/build_formal_launcher.py").read_text() and "chmod -R a-w" in (ROOT/"scripts/build_formal_launcher.py").read_text(),
        "authorization_gate": "R40_H20_EXECUTION_AUTHORIZED" in (ROOT/"formal/launch_h20.sh").read_text(),
    }
    failed = sorted(key for key, value in checks.items() if not value)
    result = {"schema_version":"forkaudit-r40-v11-local-closure-static-v1","passed":not failed,"checks":checks,"failed_checks":failed,"formal_gpu_execution":"not-run"}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
