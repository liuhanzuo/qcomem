from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    acceptance = json.loads((ROOT / "acceptance.json").read_text())
    hook = (ROOT / "executed_source/r40_real_binding_hook.py").read_text()
    verifier = (ROOT / "executed_source/r40_real_binding.py").read_text()
    tests = (ROOT / "tests/test_real_binding.py").read_text()
    for path in list((ROOT / "executed_source").glob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    checks = {
        "formal_hold": prereg["formal_gpu_execution"] == "not-run" and acceptance["formal_evidence_eligible"] is False,
        "no_formal_launcher": not any((ROOT / "formal").glob("*")),
        "reference_before_original_build": hook.index("ActualBindingVerifier(cache") < hook.index("group = original_build"),
        "actual_phase_after_original_return": hook.index("result = original_phase") < hook.index("verify_serialized_phase"),
        "actual_serializer_rows_consumed": 'gdn.get("storage_witness", {}).get("rows")' in verifier,
        "off_path_candidate_absent": "_candidate_items" not in verifier + hook,
        "four_real_fault_tests": sum(name in tests for name in ("coherent_cross_layer_swap", "request_base_alias", "post_rebind_stale", "same_geometry_one_way")) == 4,
        "selected_cell_primary_zero": '"primary_memory_hook_events": 0' in hook,
        "six_selected_coordinates": len(prereg["selected_coordinates"]) == 6,
        "three_audit_regressions_present": all(name in tests for name in ("completed_request_aliasing_incomplete_peer", "forged_normalized_serializer_storage_id", "persistent_mutation_after_prebuild_freeze")),
        "normalized_ids_from_live_keys": "actual serializer normalized storage_id/live-storage mismatch" in verifier,
        "persistent_rechecked": "persistent source content changed after freeze" in verifier,
        "no_formal_entrypoint_or_finalizer": not any((ROOT / "executed_source").glob("*entrypoint*")) and not any((ROOT / "executed_source").glob("*finalize*")),
        "blocker_machine_readable": json.loads((ROOT / "formal-blocker.json").read_text())["status"] == "hold-no-launcher",
        "exact_row_schema_gate": "serializer row exact schema drift" in verifier,
        "row_cardinality_and_unique_gate": "serializer row cardinality drift" in verifier and "serializer duplicate semantic row" in verifier,
        "full_descriptor_interval_gate": all(text in verifier for text in ("serializer stride/offset mismatch","serializer nbytes mismatch","serializer byte interval mismatch")),
        "completed_crosscheck_gate": "completed set GDN/call-args mismatch" in verifier,
        "lineage_interface_fail_closed": "lineage receipt gate unsatisfied" in verifier,
        "exact_clone_edge_binding": all(text in verifier for text in ("aten.clone.default","lineage source interval/storage mismatch","lineage destination live-object/storage mismatch")),
    }
    failed = sorted(key for key, value in checks.items() if not value)
    result = {"schema_version":"forkaudit-r40-v7-hold-static-v1","passed":not failed,"checks":checks,"failed_checks":failed,"formal_gpu_execution":"not-run"}
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
