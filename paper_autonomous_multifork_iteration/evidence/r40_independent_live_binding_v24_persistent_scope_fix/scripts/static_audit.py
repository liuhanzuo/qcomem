from __future__ import annotations

import ast
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = "HOLD_PENDING_FRESH_AUDIT_AND_H20"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    acceptance = json.loads((ROOT / "acceptance.json").read_text())
    hook = (ROOT / "executed_source/r40_real_binding_hook.py").read_text()
    verifier = (ROOT / "executed_source/r40_real_binding.py").read_text()
    tests = (ROOT / "tests/test_real_binding.py").read_text()
    lineage = (ROOT / "executed_source/r40_passive_clone_lineage.py").read_text()
    lineage_tests = (ROOT / "tests/test_strong_lineage_counterexamples.py").read_text()
    generation_tests = (ROOT / "tests/test_generation_lifecycle.py").read_text()
    launcher_tests = (ROOT / "tests/test_frozen_import_and_packaging.py").read_text()
    launcher = (ROOT / "formal/launch_h20.sh").read_text()
    launcher_builder = (ROOT / "scripts/build_formal_launcher.py").read_text()
    finalizer = (ROOT / "executed_source/r40_finalize.py").read_text()
    cuda_smoke = (ROOT / "executed_source/r40_cuda_smoke.py").read_text()
    archiver = (ROOT / "scripts/build_deterministic_archive.py").read_text()
    tree = (ROOT / "executed_source/r40_tree_closure.py").read_text()
    tree_tests = (ROOT / "tests/test_tree_closure.py").read_text()
    finalize_tests = (ROOT / "tests/test_finalize.py").read_text()
    readme = (ROOT / "README.md").read_text()
    absorbed = json.loads((ROOT / "absorbed-lineage.json").read_text())
    stage_builder = (ROOT / "scripts/stage_v6_clean.py").read_text()
    stage_tests = (ROOT / "tests/test_linux_stage_contract.py").read_text()
    clean_ledger = json.loads((ROOT / "v6-clean-members.json").read_text())
    exclusion_ledger = json.loads((ROOT / "v6-appledouble-exclusions.json").read_text())
    controlled_diff_path = ROOT / "v18-v19-controlled-diff.json"
    controlled_diff_bytes = controlled_diff_path.read_bytes()
    controlled = json.loads(controlled_diff_bytes)
    controlled_payload = dict(controlled)
    controlled_payload["payload_sha256"] = None
    controlled_payload_sha256 = hashlib.sha256(
        json.dumps(controlled_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    science_verifier = (ROOT / "scripts/verify_frozen_science.py").read_text()
    regression_runner = (ROOT / "scripts/run_linux_stage_regression.py").read_text()
    stage_regression = json.loads((ROOT / "linux-stage-regression.json").read_text())
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
        "runtime_gate_machine_readable": json.loads((ROOT / "formal-blocker.json").read_text())["status"] == STATUS,
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
        "v16_payload_identity": "forkaudit-r40-v16-real-binding-rank-v1" in hook and "forkaudit_r40_v16_real_binding_context" in hook and "r40-v4" not in hook,
        "exact_clean_memory_interface": "_run_clean_memory_cell" in hook and "_run_memory_cell" not in hook and "_run_formal_factorial_cells" in hook,
        "active_primary_fail_closed": "inside active ownership context" in hook and "primary_memory_hook_events" in hook,
        "dispatch_internal_token": "lineage event creation outside __torch_dispatch__ rejected" in lineage and "self.__dispatch_token" in lineage,
        "single_bound_capability": "lineage mode may issue only one capability" in lineage and "lineage capability verifier/group binding drift" in lineage,
        "full_live_lifecycle": "full_live_rows_verified" in verifier and "for request_index,request in enumerate(group.requests)" in verifier,
        "exact_scalar_types": "serializer exact field type drift" in verifier and "completed indices exact type drift" in verifier,
        "finalizer_rereads_actual_artifacts": "phase artifact finalizer bytes/hash drift" in finalizer and "capture lexical tree exact closure drift" in finalizer,
        "failure_ledger": "forkaudit-r40-v24-launch-failure-v1" in (ROOT/"formal/launch_h20.sh").read_text(),
        "cuda_binds_real_runner": "real runner source hash drift" in (ROOT/"executed_source/r40_cuda_smoke.py").read_text() and "_run_clean_memory_cell" in (ROOT/"executed_source/r40_cuda_smoke.py").read_text(),
        "round_robin_generation_binding": "observe_generation_step" in verifier and "_gpu_round_robin_generate" in hook and "generation round-robin schedule/order drift" in verifier,
        "eight_step_rebind_counts": "[0,1,64]" in (ROOT/"executed_source/r40_finalize.py").read_text() and "[8]*8" in (ROOT/"executed_source/r40_finalize.py").read_text(),
        "cuda_dynamic_import_registered": "sys.modules[spec.name]=runner" in (ROOT/"executed_source/r40_cuda_smoke.py").read_text(),
        "single_exit_handler": (ROOT/"formal/launch_h20.sh").read_text().count("trap on_exit EXIT") == 1 and "trap 'rm" not in (ROOT/"formal/launch_h20.sh").read_text(),
        "formal_zero_skip_preflight": (ROOT/"executed_source/r40_formal_preflight.py").is_file() and "result.skipped" in (ROOT/"executed_source/r40_formal_preflight.py").read_text() and "r40_formal_preflight.py" in (ROOT/"scripts/build_formal_launcher.py").read_text(),
        "authorizing_exact_types": "authorizing flag/type drift" in (ROOT/"executed_source/r40_finalize.py").read_text() and "absence authorizing type/count drift" in (ROOT/"executed_source/r40_finalize.py").read_text(),
        "explicit_dual_authorization": '${R40_H20_EXECUTION_AUTHORIZED:?}' not in launcher and '== yes' in launcher and "R40_V24_FRESH_AUDIT_APPROVED" in launcher,
        "formal_preflight_control_environment_isolated": "isolated_launcher_env" in launcher_tests and "dict(os.environ" not in launcher_tests and all(name in launcher_tests for name in ("R40_H20_EXECUTION_AUTHORIZED","R40_V19_FRESH_AUDIT_APPROVED","R40_V23_FRESH_AUDIT_APPROVED","R40_V24_FRESH_AUDIT_APPROVED","R40_ONE_SHOT_MARKER","R40_LAUNCHER_HANDLER_SELFTEST","R40_LAUNCHER_ATOMIC_GATE_SELFTEST","R40_SELFTEST_RESULT_ACTION","R40_SELFTEST_SIGNAL_HOLD")),
        "all_non_target_isolation": "non-target request changed before its scheduled call" in verifier and "per_call_isolation_verified" in verifier,
        "global_live_nonalias": "callback live ownership alias across persistent/peer/same-request coordinates" in verifier,
        "selected_and_terminal_primary_counts": 'primary_memory_calls_observed"]==7' in (ROOT/"executed_source/r40_finalize.py").read_text() and 'primary_memory_calls_observed")==expected_calls' in (ROOT/"executed_source/r40_finalize.py").read_text(),
        "exact_capture_tree_closure": "lexical_tree(root)" in finalizer and "capture lexical tree exact closure drift" in finalizer,
        "artifact_rank_cell_phase_binding": "wrapper/GDN rank-cell-phase binding drift" in (ROOT/"executed_source/r40_finalize.py").read_text() and "phase artifact path reused" in (ROOT/"executed_source/r40_finalize.py").read_text(),
        "lineage_unchanged_gate": "source_values_rechecked_unchanged" in (ROOT/"executed_source/r40_finalize.py").read_text(),
        "primary_call_coverage_exact": prereg["acceptance"]["primary_memory_calls_per_rank"] == 12 and "expected_primary_memory_calls" in (ROOT/"executed_source/r40_rank_entrypoint.py").read_text() and "primary_call_coverage_proof" in (ROOT/"executed_source/r40_finalize.py").read_text(),
        "artifact_reread_and_cleanup": "_bind_phase_artifact" in hook and "artifact_path.unlink" in hook,
        "cuda_smoke_before_science": "r40_cuda_smoke.py" in (ROOT/"scripts/build_formal_launcher.py").read_text(),
        "linux_sha_readonly_terminal": "sha256sum -c" in (ROOT/"scripts/build_formal_launcher.py").read_text() and "chmod -R a-w" in (ROOT/"scripts/build_formal_launcher.py").read_text(),
        "authorization_gate": "R40_H20_EXECUTION_AUTHORIZED" in (ROOT/"formal/launch_h20.sh").read_text(),
        "global_freshness_against_full_history": "functional rebind endpoint is not globally fresh" in verifier and "prior_object_ids" in verifier and "prior_storage" in verifier,
        "rotation_permutation_regressions": "rotation_and_permutation_of_preexisting_target_objects_rejected" in generation_tests,
        "full_endpoint_snapshot_stability": all(x in verifier for x in ("persistent full descriptor/interval changed after freeze","non-target request changed before its scheduled call","functional rebind descriptor/offset/interval unauthorized")),
        "signal_failure_ledger_regressions": 'for signal_name in ("SIGTERM","SIGINT")' in launcher_tests and "preexec_fn=restore_launcher_signal_dispositions" in launcher_tests and all(name in launcher_tests for name in ("signal.SIGHUP","signal.SIGINT","signal.SIGQUIT","signal.SIGTERM","signal.SIG_DFL")) and "R40_SELFTEST_SIGNAL_HOLD" in launcher_tests + launcher and "while :; do :; done" in launcher and "trap 'exit 130' INT" in launcher and "trap 'exit 143' TERM" in launcher,
        "atomic_one_shot_no_toctou": "mkdir -- \"$MARKER\"" in launcher and "[[ ! -e \"$MARKER\" ]]" not in launcher and "test_atomic_one_shot_concurrency" in launcher_tests,
        "external_operator_hash_approval": all(x in launcher for x in ("R40_V24_APPROVED_SOURCE_LEDGER_SHA256","R40_V24_APPROVED_ARCHIVE_SHA256","operator-approved v24 source ledger mismatch","operator-approved v24 overlay archive mismatch")),
        "launcher_v6_single_stable_descriptor_snapshot": all(x in launcher_builder for x in ("os.O_NOFOLLOW","os.fstat(descriptor)","snapshot = read_regular_snapshot(args.v6)","hashlib.sha256(snapshot)","snapshot.decode(\"utf-8\", errors=\"strict\")","value = transform(source)")) and launcher_builder.count("read_regular_snapshot(args.v6)") == 1 and "args.v6.read_" not in launcher_builder,
        "launcher_v6_pathname_swap_regression": all(x in launcher_tests for x in ("swapping_open","os.replace(attacker,source)","self.assertEqual(opened,[source])","self.assertFalse(output.exists())")),
        "launcher_v6_strict_utf8_regression": "self.assertRaises(UnicodeDecodeError)" in launcher_tests and "source.write_bytes(b\"\\xff\")" in launcher_tests,
        "deterministic_safe_archive": all(x in archiver for x in ("gzip.GzipFile","mtime=0","tarfile.USTAR_FORMAT","info.uid = info.gid = 0","info.pax_headers = {}","output.open(\"xb\")")),
        "archive_rejects_special_and_link_nodes": all(x in archiver for x in ("stat.S_ISLNK","archive hardlink forbidden","archive special node forbidden","O_NOFOLLOW")) and "safe_archive_rejects_symlink_hardlink_and_fifo_without_output" in launcher_tests,
        "archive_output_normalized_contained_prewrite": all(x in archiver for x in ("archive output contains forbidden dotdot component","archive output must be strictly inside output root","archive output parent canonical containment drift")) and "archive_dotdot_output_has_no_outside_side_effect" in launcher_tests,
        "archive_command_determinism_regression": "deterministic_archive_rebuild_is_byte_identical_and_nonoverwriting" in launcher_tests,
        "finalizer_exact_phase_order": "phase receipt/order mismatch" in finalizer,
        "finalizer_exact_schema_no_extras": "rank artifact exact schema drift" in finalizer and "lineage geometry/schema drift" in finalizer,
        "finalizer_symlink_and_tree_closure": "lexical_tree(root)" in finalizer and "capture/terminal tree symlink forbidden" in tree,
        "cumulative_historical_identity_universe": all(x in verifier for x in ("historical_tensors","historical_object_ids","historical_storage_keys","prior_object_ids=set(self.historical_object_ids)")),
        "strong_refs_prevent_allocator_identity_reuse": "Strong references make Python object IDs" in verifier and "self.historical_tensors.append(current)" in verifier,
        "superseded_cross_request_regressions": all(x in generation_tests for x in ("superseded_historical_endpoints_cannot_move_to_next_request","storage-view","cyclic")),
        "lstat_scandir_all_node_types": all(x in tree for x in ("os.lstat","os.scandir","stat.S_ISLNK","stat.S_ISDIR","stat.S_ISREG","st_nlink")) and all(x in tree_tests for x in ("mkfifo","AF_UNIX","os.link","symlink_to")),
        "terminal_lexical_full_tree_ledger": "r40_tree_closure.py" in (ROOT/"scripts/build_formal_launcher.py").read_text() and "terminal-tree.json" in (ROOT/"scripts/build_formal_launcher.py").read_text() and "return s.replace(old, new)" in (ROOT/"scripts/build_formal_launcher.py").read_text(),
        "terminal_output_exclusive_no_special_preexistence": "os.path.lexists(lexical)" in tree and 'path.open("xb")' in tree and "all_preexisting_terminal_output_node_types_fail_closed" in tree_tests,
        "all_declared_statuses_remain_hold": all(
            json.loads((ROOT / name).read_text()).get("status") == STATUS
            for name in ("preregistration.json", "formal-blocker.json", "acceptance.json", "absorbed-lineage.json", "package-manifest.json")
        ),
        "root_lstat_precedes_resolve": tree.index("metadata = os.lstat(lexical)") < tree.index("canonical = lexical.resolve(strict=True)"),
        "root_canonical_equals_lexical": "canonical != lexical" in tree and "canonical tree root differs from lexical absolute root" in tree,
        "root_symlink_command_regression": "root_symlink_and_symlinked_parent_rejected_before_resolve_or_write" in tree_tests,
        "output_dotdot_rejected_prewrite": 'if ".." in raw.parts' in tree and "output path contains forbidden dotdot component" in tree,
        "output_strict_root_containment": "output path must be strictly inside tree root" in tree and "output parent canonical containment drift" in tree,
        "dotdot_no_side_effect_command_regression": "dotdot_output_rejected_by_command_with_no_outside_side_effect" in tree_tests and "outside.exists()" in tree_tests,
        "exact_expected_path_whitelist": "expected_paths" in tree and "terminal path exact whitelist drift" in tree and "existing terminal path exact whitelist drift" in tree and "formal result exact expected path whitelist drift" in tree,
        "predetermined_formal_path_authorities": all(text in tree for text in ("PREFLIGHT_EXACT_PATHS","PRIMARY_EXACT_LOGS","PRIMARY_EXACT_STAGES","primary/scientific-artifacts.sha256","private-model-view-manifest.json","formal-binding/terminal-files.sha256","_expected_capture_paths")),
        "fixed_invocation_paths_exact_schema_and_receipt_binding": all(text in tree for text in (
            "PRIMARY_INVOCATION_FIELDS",
            'f"compiled-dispatch-capture/rank-{rank}/invocation.json"',
            "invocation_bytes == _canonical_json_bytes(invocation)",
            'invocation["schema_version"] == "forkaudit-r39-primary-rank-invocation-v2"',
            'invocation["runner_sha256"] == binding.get("runner_sha256")',
            'invocation["runner_argv"] == binding.get("runner_argv")',
            'invocation["primary_shard_sha256"] == binding.get("primary_shard_sha256")',
            'argv_sha256 == binding.get("runner_argv_sha256")',
            "expected.update({capture_receipt, invocation_relative})",
        )),
        "producer_accurate_invocation_fixture_and_fail_closed_regressions": all(text in tree_tests for text in (
            '"execution_binding"',
            '"forkaudit-r39-primary-rank-invocation-v2"',
            'f"compiled-dispatch-capture/rank-{rank}/invocation.json" for rank in range(8)',
            "test_formal_invocation_schema_and_receipt_binding_fail_closed",
            '("missing", None)',
            '"extra-field"',
            '"schema-version"',
            '"runner-sha"',
            '"runner-argv"',
            '"shard-sha"',
        )),
        "exact_node_file_directory_counts": all(text in tree for text in ("expected_node_count","expected_regular_file_count","expected_directory_count","final_node_count","final_regular_file_count","final_directory_count")),
        "per_file_exact_content_schema": "content_schema" in tree and "terminal file bytes/hash/exact schema drift" in tree,
        "extra_regular_and_directory_regressions": all(text in tree_tests for text in ("extra_regular_file_fails_exact_whitelist","extra_directory_fails_exact_whitelist")),
        "preprepare_extra_regular_and_directory_regression": "formal_prepare_rejects_preexisting_extra_regular_and_directory" in tree_tests and "expected-paths" in tree_tests,
        "formal_predetermined_plan_success_regression": "formal_predetermined_path_plan_and_terminal_close_succeed" in tree_tests,
        "exact_schema_and_count_regressions": "resealed_count_and_per_file_schema_tamper_fail_closed" in tree_tests and "terminal_ledger_has_exact_whitelist_counts_and_per_file_schema" in tree_tests,
        "terminal_closure_exclusive_command_regression": "terminal_closure_command_is_exclusive_nonoverwrite" in tree_tests,
        "complete_exclusive_command_regression": "complete_command_is_exclusive_nonoverwrite" in tree_tests,
        "terminal_tree_exclusive_command_regression": "terminal_tree_command_is_exclusive_nonoverwrite" in tree_tests,
        "aggregate_exclusive_command_regression": "aggregate_command_refuses_preexisting_terminal_output_without_mutation" in finalize_tests and "publish_json_exclusive" in finalizer,
        "cuda_exclusive_command_regression": "cuda_smoke_command_refuses_preexisting_terminal_output_without_gpu" in launcher_tests and "publish_json_exclusive" in cuda_smoke,
        "launcher_uses_exclusive_terminal_commands": all(text in launcher_builder for text in ('r40_tree_closure.py" prepare','r40_tree_closure.py" complete','r40_tree_closure.py" close','--terminal-root "$RESULT_ROOT"','r40_cuda_smoke.py" --root "$RESULT_ROOT"')),
        "launcher_explicit_exact_existing_paths": all(text in launcher_builder for text in ('r40_tree_closure.py" expected-paths','R40_EXPECTED_PATH_ARGS+=(--expected-existing-path','"${R40_EXPECTED_PATH_ARGS[@]}"')) and "R40_EXPECTED_PATH_ARGS+=(--expected-existing-path" in launcher_tests,
        "launcher_redirection_counterexample_regression": "transformed_launcher_uses_only_exclusive_r40_terminal_publications" in launcher_tests and 'self.assertNotIn("touch' in launcher_tests and 'self.assertNotIn("> ' in launcher_tests,
        "all_r40_terminal_json_uses_exclusive_primitive": "publish_json_exclusive" in finalizer and "publish_json_exclusive" in cuda_smoke and "publish_json_exclusive(canonical_root, output_path" in tree,
        "producer_fix_and_persistent_scope_documented": all(text in readme for text in ("_prepare_request_gdn_base","60 GDN tensors","direct destination","Borrow delegates","_convert_persistent","exact backbone and","unmarked cache outside")),
        "v24_pending_and_not_h20_evidence_status_hold": "Status: **HOLD_PENDING_FRESH_AUDIT_AND_H20**." in readme and "not H20 evidence" in readme and "No v24 GPU result marker" in readme,
        "canonical_v6_archive_exact_sha": "306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82" in stage_builder and "R40_V24_APPROVED_V6_ARCHIVE_SHA256" in launcher,
        "cross_platform_exact_member_partition": all(text in stage_builder for text in ("EXPECTED_ARCHIVE_MEMBERS = 260","EXPECTED_RETAINED_MEMBERS = 130","EXPECTED_EXCLUDED_MEMBERS = 130")) and clean_ledger["retained_member_count"] == 130 and exclusion_ledger["excluded_member_count"] == 130,
        "appledouble_exact_proof": all(text in stage_builder for text in ("APPLEDOUBLE_MAGIC = 0x00051607","APPLEDOUBLE_VERSION = 0x00020000","EXPECTED_APPLEDOUBLE_SHA256","entry descriptors drift")),
        "exclusion_only_basename_rule": "basename.startswith(\"._\")" in stage_builder and exclusion_ledger["exclusion_rule"] == "basename-starts-with-._-and-exact-AppleDouble-proof",
        "all_exclusions_have_companions_no_science_excluded": exclusion_ledger["all_exclusions_have_retained_logical_companions"] is True and exclusion_ledger["scientific_files_excluded"] == 0 and len(clean_ledger["required_science_paths"]) == 18,
        "retained_byte_mode_type_frozen": all(key in clean_ledger["rows"][0] for key in ("path","type","mode","size","sha256")) and "final stage byte/mode/type drift" in stage_builder,
        "unsafe_link_special_rejected": all(text in stage_builder for text in ("escapes staging root","link/special member forbidden","symlink forbidden","special node forbidden")) and "archive_counterexamples_fail_closed" in stage_tests,
        "prepare_is_atomic_nonoverwriting": all(text in stage_builder for text in ("output already exists or is a special node","private staging temporary path collision","rename_directory_noreplace(temporary, output)")) and "prepare_is_nonoverwriting" in stage_tests,
        "atomic_publication_is_kernel_noreplace": all(text in stage_builder for text in ("renameat2","renamex_np","stage output appeared before atomic no-replace publication")) and "atomic_directory_publication_refuses_even_empty_existing_target" in stage_tests,
        "prepare_dotdot_zero_side_effect": "stage output contains forbidden dotdot component" in stage_builder and "prepare_dotdot_and_preexisting_nodes_have_zero_side_effect" in stage_tests,
        "stage_root_symlink_and_symlinked_parent_regressions": "root_symlink_and_symlinked_parent_fail_before_stage_action" in stage_tests and stage_builder.index("metadata = os.lstat(lexical)") < stage_builder.index("lexical.resolve(strict=True)"),
        "stage_exact_path_whitelist": "final stage exact path whitelist drift" in stage_builder and "lexical_stage_tree" in stage_builder,
        "stage_receipt_exact_canonical_bytes_and_mode": "canonical JSON byte encoding drift" in stage_builder and "stage receipt mode drift" in stage_builder and "receipt-mode" in stage_tests,
        "overlay_unsafe_link_special_counterexamples": "overlay_archive_counterexamples_fail_closed" in stage_tests and "v24 overlay link/special member forbidden" in stage_builder,
        "repaired_ledgers_resealed_tamper_rejected": "self_consistently_resealed_clean_and_exclusion_ledger_tamper_fails" in stage_tests and "differs from canonical archive" in stage_builder,
        "zero_appledouble_before_result_action": launcher.index("stage_v6_clean.py\" verify") < launcher.index("MARKER=${R40_ONE_SHOT_MARKER") and launcher.index("-name '._*'") < launcher.index("MARKER=${R40_ONE_SHOT_MARKER"),
        "fixed_v24_stage_and_result_names": "qcomem_r40_v24_persistent_scope_fix_20260831a" in launcher_builder + launcher and "r40-v24-persistent-scope-fix-20260831a" in launcher_builder,
        "v18_v19_controlled_payload_diff_exact": controlled["current_payload_file_count"] == 10 and controlled["byte_identical_file_count"] == 9 and controlled["controlled_change_file_count"] == 1 and len(controlled["rows"]) == 10 and sum(row["byte_identical"] is True for row in controlled["rows"]) == 9 and all(row["v18_sha256"] == row["v19_sha256"] and row["v18_size"] == row["v19_size"] and row["classification"] == "byte-identical" for row in controlled["rows"] if row["byte_identical"] is True) and [(row["path"], row["classification"]) for row in controlled["rows"] if row["byte_identical"] is False] == [("executed_source/r40_tree_closure.py", "terminal-governance-invocation-closure-repair")],
        "controlled_diff_external_file_and_payload_pins": hashlib.sha256(controlled_diff_bytes).hexdigest() == "fc81a6a0b6cbac46708ef4f9680db9829ba7c2e22f437d9e8f8629be81c76f9c" and controlled["payload_sha256"] == controlled_payload_sha256 == "73fae1086b003a4233546b6d6c179bf0cac4e44445efb790e9aa54801d44b6f6",
        "self_contained_current_payload_and_controlled_diff_pins": hashlib.sha256((ROOT/"v24-current-payload.sha256").read_bytes()).hexdigest() == "4eba13c77e5ecc1967509b5efd03c0b1e8316f92a4ec6860f3cdbf22ed18db51" and hashlib.sha256((ROOT/"v23-v24-controlled-diff.json").read_bytes()).hexdigest() == "e25c9a2af00023d11330ee722bbe9c9b2501900ecf648a9a90eff5e510228bbd" and all(text in science_verifier for text in ("EXPECTED_MANIFEST_SHA256","EXPECTED_DIFF_SHA256","EXPECTED_DIFF_PAYLOAD_SHA256","sibling_source_accessed")),
        "self_contained_current_payload_hash_size_seal": all(text in science_verifier for text in ("v24 payload hash/size drift","executed-source exact file set drift","v23-v24 controlled diff seal drift")) and "verify_scientific_payload(root=ROOT" in stage_tests,
        "external_runner_builder_unchanged": controlled["immutable_external_runner_sha256"] == "9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775" and controlled["immutable_external_builder_sha256"] == "546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e",
        "generated_science_launcher_only_identifiers_changed": controlled["generated_launcher_diff_is_only_stage_result_package_identifiers"] is True and controlled["v18_generated_launcher_sha256"] == "8d5ba77f9b61b760346334b4bca041e1ac0176719c5b8bd2e616a29b24226636" and controlled["v19_generated_launcher_sha256"] == "ef1f68028fbec4180925c60701ed6d850975c0eeeff541964cab60fafa2e20ed",
        "linux_raw_extraction_regression_present": "python_gnu_equivalent_raw_extraction_materializes_v16_blocker" in stage_tests and "130" in stage_tests,
        "clean_162_formal_zero_skip_gate_pending_v24_replay_recorded": stage_regression["clean_stage"]["formal_required_tests"] == 162 and stage_regression["clean_stage"]["formal_required_zero_skip"] is True and stage_regression["clean_stage"]["status"] == "pending-v24-replay",
        "executable_same_python_raw_clean_162_regression": all(text in regression_runner for text in ("raw regression did not run 162 tests","clean regression did not run 162 tests","clean regression requires zero skip","raw-162.log","clean-162.log")),
        "clean_stage_formal_preflight_94_zero_skip_regression": all(text in regression_runner for text in ("clean-formal-preflight-94.log","clean formal preflight did not run exactly 94 package tests","clean formal preflight requires zero skip","v18_sibling_absent","R40_V24_OVERLAY_ARCHIVE")),
    }
    failed = sorted(key for key, value in checks.items() if not value)
    result = {"schema_version":"forkaudit-r40-v24-persistent-scope-fix-static-v1","status":STATUS,"all_checks_satisfied":not failed,"checks":checks,"failed_checks":failed,"formal_gpu_execution":"not-run"}
    encoded = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if args.output is None:
        print(encoded.decode("utf-8"), end="")
    else:
        with args.output.open("xb") as stream:
            stream.write(encoded)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
