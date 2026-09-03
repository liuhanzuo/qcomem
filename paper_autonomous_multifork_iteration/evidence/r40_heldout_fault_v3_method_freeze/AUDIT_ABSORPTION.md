# Method-v2 audit absorption matrix

All rows below are local engineering counterexamples.  Passing them does not
create a held-out result.

| ID | Audit counterexample | Frozen v3 response | Negative test |
|---|---|---|---|
| A01 | Caller changes geometry, policy, model, schedule, or hashes. | Zero-argument authoritative loader plus compiled file hashes and exact schema. | `test_authoritative_geometry_schedule_model_and_hash_mutations_fail` |
| A02 | Wrong request, call order, or per-call input-token count is accepted. | Exact sixteen-row schedule authority. | `test_schedule_wrong_q_request_or_order_fails` |
| A03 | Verifier consumes curated receipt mappings and omits disk artifacts. | Zero-argument fixed-root entry point and exact lane directory/file enumeration. | `test_zero_argument_public_authority_verifier_and_executor`; `test_extra_missing_symlink_and_tampered_sidecar_fail` |
| A04 | Receipt is replayed across campaign, run, lane, case, GPU, method, or call. | Every receipt and lane binding carries all frozen identities and hashes. | `test_receipt_campaign_lane_fault_gpu_schedule_and_method_bindings_fail` |
| A05 | Model callback self-reports a favorable state. | Wrapper accepts only token and logit tensor; state comes from bound live tensors after wrapper-owned synchronization. | `test_model_result_cannot_supply_or_append_state_mapping`; `test_wrapper_synchronizes_and_reads_mutated_live_tensors_not_model_state` |
| A06 | Tensor enumeration order changes the digest. | Canonical role-sorted metadata and raw-byte framing. | `test_component_digest_is_canonical_over_role_order` |
| A07 | Candidate stays stale or rolls back while reference advances. | Exact paired pre/post KV/GDN content, length, version, and epoch. | `test_reference_changes_candidate_rollback_fails_structural_and_atomic`; `test_gdn_content_only_drift_fails_paired_structural_gate` |
| A08 | Observation/event IDs are reused across lanes or allocator records. | One global uniqueness set for the complete campaign. | `test_observation_and_sync_ids_are_global_not_per_lane`; `test_allocator_peak_monotonicity_binding_and_global_event_uniqueness` |
| A09 | Allocator peak decreases or endpoint identity is transplanted. | Monotone peak plus campaign/run/lane/case/GPU/device binding. | `test_allocator_peak_monotonicity_binding_and_global_event_uniqueness` |
| A10 | Changed output root or config bypasses a one-shot lock. | Hardcoded campaign parent, sealed sole output, campaign-global and config-hash O_EXCL locks retained forever. | `test_campaign_global_lock_blocks_changed_config_or_output_root`; formal binding mutation tests |
| A11 | Wrong, busy, or partially specified node starts. | Exact eight ordered UUIDs, H20 family, idle-memory bound, and empty compute-process query before locks. | `test_exact_empty_specified_eight_h20_preflight` |
| A12 | Interrupt/crash leaves selected cases without an auditable terminal. | Eight pending terminals before workers and one idempotent finalizer for signals/exceptions/exit with process-group kill. | `test_all_eight_pending_terminals_exist_before_workers`; `test_single_idempotent_finalizer_kills_groups_and_writes_all_terminals` |
| A13 | Source or formal config changes between preflight and completion. | Fixed source/config rehash before and after execution. | `test_method_manifest_member_tamper_fails`; `test_pre_post_rehash_rejects_formal_config_drift` |
| A14 | Full-vocabulary sidecar is truncated, tampered, symlinked, or nonfinite. | Exact path, count, byte size, SHA-256, shape/dtype, and finiteness reread from disk. | `test_extra_missing_symlink_and_tampered_sidecar_fail`; `test_nonfinite_complete_logits_fail_even_when_hash_matches` |
| A15 | Only a partial set of cases/lanes is verified. | Exact eight-case × three-lane campaign-directory enumeration. | `test_complete_eight_case_three_lane_campaign_is_disk_enumerated` |

