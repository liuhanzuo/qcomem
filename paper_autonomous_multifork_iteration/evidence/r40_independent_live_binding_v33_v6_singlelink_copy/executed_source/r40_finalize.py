from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from r40_tree_closure import AGGREGATE_FIELDS, ensure_output_absent, lexical_tree, publish_json_exclusive, validate_root


def req(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_sha(value):
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def seal_ok(value):
    observed = value.get("payload_sha256")
    candidate = dict(value); candidate["payload_sha256"] = None
    return is_sha(observed) and hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == observed


def ledger_seal_ok(value):
    observed = value.get("ledger_sha256")
    candidate = dict(value); candidate["ledger_sha256"] = None
    return is_sha(observed) and hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == observed


def validate_rank_hook_counters(counters):
    expected = {"selected_builds": 1, "selected_phases": 3, "primary_memory_calls_observed": 8, "primary_memory_hook_events": 0}
    req(type(counters) is dict and set(counters) == set(expected), "hook counter schema drift")
    req(all(type(item) is int for item in counters.values()) and counters == expected, "hook counter drift")


def validate_functional_ledger(ledger, source_coordinates):
    fields = {"schema_version", "call_count", "edge_count", "edges_per_call", "all_new_tensor_objects", "all_new_storages", "all_descriptors_authorized", "all_contents_recorded", "calls", "ledger_sha256"}
    req(type(ledger) is dict and set(ledger) == fields and ledger_seal_ok(ledger), "functional ledger schema/seal drift")
    req(ledger["schema_version"] == "forkaudit-r40-v32-functional-rebind-ledger-v1", "functional ledger identity drift")
    req(all(ledger[key] is True for key in ("all_new_tensor_objects", "all_new_storages", "all_descriptors_authorized", "all_contents_recorded")), "functional ledger predicate drift")
    req((ledger["call_count"], ledger["edge_count"], ledger["edges_per_call"]) == (64, 3840, 60), "functional ledger count drift")
    calls = ledger["calls"]
    req(type(calls) is list and len(calls) == 64, "functional call ledger cardinality drift")
    call_fields = {"call_index", "round_index", "request_index", "request_version", "edge_count", "edges", "completed_request_indices_after_call", "private_request_rows_after_call", "borrowed_request_rows_after_call", "target_all_new", "non_target_unchanged", "persistent_unchanged", "completed_private", "incomplete_exact_alias"}
    edge_fields = {"coordinate", "version", "pre", "post", "new_tensor_object", "new_storage", "descriptor_authorized", "content_recorded"}
    snapshot_fields = {"object_id", "storage_key", "descriptor", "content_sha256"}
    descriptor_fields = {"shape", "stride", "storage_offset", "dtype", "device", "storage_nbytes", "tensor_nbytes", "byte_interval"}
    expected_coordinates = [[layer, family, 0] for layer in range(source_coordinates // 2) for family in ("conv", "recurrent")]
    post_objects = set(); post_storages = set()
    for index, call in enumerate(calls):
        req(type(call) is dict and set(call) == call_fields, "functional call schema drift")
        request_index = index % 8; round_index = index // 8
        completed = list(range(request_index + 1)) if round_index == 0 else list(range(8))
        req((call["call_index"], call["round_index"], call["request_index"], call["request_version"], call["edge_count"]) == (index, round_index, request_index, round_index + 1, 60), "functional call schedule/count drift")
        req(call["completed_request_indices_after_call"] == completed and call["private_request_rows_after_call"] == len(completed) * 60 and call["borrowed_request_rows_after_call"] == (8 - len(completed)) * 60, "functional call ownership vector drift")
        req(all(call[key] is True for key in ("target_all_new", "non_target_unchanged", "persistent_unchanged", "completed_private", "incomplete_exact_alias")), "functional call predicate drift")
        req(type(call["edges"]) is list and len(call["edges"]) == 60 and [edge.get("coordinate") for edge in call["edges"]] == expected_coordinates, "functional edge coordinate universe/order drift")
        for edge in call["edges"]:
            req(type(edge) is dict and set(edge) == edge_fields and edge["version"] == round_index + 1, "functional edge schema/version drift")
            req(all(edge[key] is True for key in ("new_tensor_object", "new_storage", "descriptor_authorized", "content_recorded")), "functional edge predicate drift")
            for snapshot in (edge["pre"], edge["post"]):
                req(type(snapshot) is dict and set(snapshot) == snapshot_fields and type(snapshot["object_id"]) is int and type(snapshot["storage_key"]) is list and len(snapshot["storage_key"]) == 3 and type(snapshot["storage_key"][0]) is str and all(type(value) is int for value in snapshot["storage_key"][1:]) and is_sha(snapshot["content_sha256"]), "functional endpoint identity/content drift")
                descriptor = snapshot["descriptor"]
                req(type(descriptor) is dict and set(descriptor) == descriptor_fields and type(descriptor["shape"]) is list and type(descriptor["stride"]) is list and type(descriptor["byte_interval"]) is list and all(type(value) is int for value in descriptor["shape"] + descriptor["stride"] + descriptor["byte_interval"]) and all(type(descriptor[key]) is int for key in ("storage_offset", "storage_nbytes", "tensor_nbytes")) and type(descriptor["dtype"]) is str and type(descriptor["device"]) is str, "functional endpoint descriptor drift")
            req(edge["post"]["object_id"] != edge["pre"]["object_id"] and edge["post"]["storage_key"] != edge["pre"]["storage_key"], "functional edge is not new")
            post_storage = tuple(edge["post"]["storage_key"])
            req(edge["post"]["object_id"] not in post_objects and post_storage not in post_storages, "functional post endpoint reused across ledger")
            post_objects.add(edge["post"]["object_id"]); post_storages.add(post_storage)
    return len(calls), sum(len(call["edges"]) for call in calls)


def finalize(root, pre, ranks=8):
    observed = lexical_tree(root); expected = {}
    for rank in range(ranks):
        expected[f"rank-{rank}"] = {"kind": "directory"}; expected[f"rank-{rank}/raw"] = {"kind": "directory"}
        for name in ("real-binding.json", "global-absence.json"):
            relative = f"rank-{rank}/raw/{name}"; data = (root / relative).read_bytes() if (root / relative).exists() else b""
            expected[relative] = {"kind": "regular", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    req(observed == expected, "capture lexical tree exact closure drift")
    results = []; primary = []; bindings = None; artifact_paths = set()
    rank_fields = {"schema_version", "experiment_id", "rank", "selected_cell", "phase_order", "phase_receipts", "source_reference_coordinate_count", "actual_selected_rows_verified", "actual_storage_rows_verified", "count_vector", "real_builder_verified", "actual_phase_serializer_verified", "off_path_candidate_detector_used", "producer_coverage", "primary_memory_hook_events", "global_hook_counters", "execution_bindings", "lineage_summary", "lineage_summary_sha256", "lineage_receipt", "functional_rebind_ledger", "formal_gpu_execution", "payload_sha256"}
    phase_fields = {"phase", "selected_rows_verified", "full_live_rows_verified", "generation_calls_verified", "functional_rebind_edges_verified", "request_rebind_counts", "per_call_isolation_verified", "private_request_rows_verified", "borrowed_request_rows_verified", "actual_storage_rows_verified", "actual_serializer_compared", "artifact_relative_path", "artifact_sha256", "artifact_bytes", "gdn_sha256"}
    for rank in range(ranks):
        binding = root / f"rank-{rank}/raw/real-binding.json"; absence = root / f"rank-{rank}/raw/global-absence.json"
        req(binding.is_file() and absence.is_file(), "rank artifact missing")
        value = json.loads(binding.read_text()); zero = json.loads(absence.read_text())
        req(seal_ok(value) and seal_ok(zero), "rank/absence seal drift")
        req(set(value) == rank_fields and value["schema_version"] == "forkaudit-r40-v32-borrowed-transition-rank-v1" and value["experiment_id"] == pre["experiment_id"], "rank artifact exact schema/identity drift")
        req(set(zero) == {"schema_version", "rank", "selected_builds", "selected_phases", "primary_memory_calls_observed", "primary_memory_hook_events", "expected_primary_memory_calls", "primary_call_coverage_proof", "primary_absence_proof", "payload_sha256"}, "absence exact schema drift")
        req(type(value["rank"]) is int and value["rank"] == rank and value["selected_cell"] == pre["selected_cell"], "rank/cell drift")
        req(value["real_builder_verified"] is True and value["actual_phase_serializer_verified"] is True and value["off_path_candidate_detector_used"] is False and type(value["primary_memory_hook_events"]) is int and value["primary_memory_hook_events"] == 0, "authorizing flag/type drift")
        coverage = value["producer_coverage"]
        req(set(coverage) == {"prebuild_reference_frozen", "real_group_observed", "borrowed_setup_exact_aliases_observed", "functional_rebind_endpoints_observed", "actual_serializer_rows_observed", "persistent_rechecked_each_phase", "all_storage_rows_normalized_against_live_keys"} and all(item is True for item in coverage.values()), "producer coverage drift")
        counters = value["global_hook_counters"]
        validate_rank_hook_counters(counters)
        vector = value["count_vector"]
        req(vector == {"source_reference_coordinates": 60, "selected_rows_by_phase": [6, 6, 6], "storage_rows_by_phase": [540, 540, 540], "full_live_rows_by_phase": [540, 540, 540], "generation_calls_by_phase": [0, 1, 64], "functional_rebind_edges_by_phase": [0, 60, 3840], "request_rebind_counts_by_phase": [[0] * 8, [1, 0, 0, 0, 0, 0, 0, 0], [8] * 8], "private_request_rows_by_phase": [0, 60, 480], "borrowed_request_rows_by_phase": [480, 420, 0], "primary_memory_hook_events": 0}, "count vector drift")
        req(value["phase_order"] == ["setup_pre_transition", "post_transition", "post_generation"] and value["source_reference_coordinate_count"] == 60 and value["actual_selected_rows_verified"] == 18 and value["actual_storage_rows_verified"] == 1620, "phase/source/count drift")
        summary = value["lineage_summary"]
        req(set(summary) == {"policy", "request_count", "source_coordinate_count", "captured_lineage_edges", "all_exact_expected_source_aliases", "source_values_rechecked_unchanged"} and summary == {"policy": "borrowed", "request_count": 8, "source_coordinate_count": 60, "captured_lineage_edges": 0, "all_exact_expected_source_aliases": True, "source_values_rechecked_unchanged": True}, "borrowed lineage summary drift")
        req(value["lineage_summary_sha256"] == hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "lineage summary hash drift")
        lineage = value["lineage_receipt"]
        req(lineage == {"opaque_capability_consumed": True, "binding_policy": "borrowed-exact-persistent-alias", "selected_binding_count": 5, "selected_exact_alias_count": 5, "selected_clone_edge_count": 0}, "selected borrowed lineage receipt drift")
        calls, edges = validate_functional_ledger(value["functional_rebind_ledger"], 60)
        req(set(value["execution_bindings"]) == set(pre["required_execution_binding_fields"]) and all(is_sha(item) for item in value["execution_bindings"].values()), "execution binding fields/SHA drift")
        if bindings is None: bindings = value["execution_bindings"]
        req(value["execution_bindings"] == bindings, "cross-rank execution binding drift")
        receipts = value["phase_receipts"]
        req(type(receipts) is list and len(receipts) == 3 and [item.get("phase") for item in receipts] == value["phase_order"], "phase receipt/order mismatch")
        for phase_index, receipt in enumerate(receipts):
            req(set(receipt) == phase_fields and receipt["selected_rows_verified"] == 6 and receipt["full_live_rows_verified"] == 540 and receipt["actual_storage_rows_verified"] == 540 and receipt["actual_serializer_compared"] is True and receipt["per_call_isolation_verified"] is True, "phase row/schema drift")
            req((receipt["generation_calls_verified"], receipt["functional_rebind_edges_verified"], receipt["private_request_rows_verified"], receipt["borrowed_request_rows_verified"]) == ([0, 1, 64][phase_index], [0, 60, 3840][phase_index], [0, 60, 480][phase_index], [480, 420, 0][phase_index]), "phase ownership/rebind vector drift")
            req(receipt["request_rebind_counts"] == [[0] * 8, [1, 0, 0, 0, 0, 0, 0, 0], [8] * 8][phase_index], "phase request rebind count drift")
            relative = receipt["artifact_relative_path"]
            req(type(relative) is str and relative not in artifact_paths and is_sha(receipt["artifact_sha256"]) and is_sha(receipt["gdn_sha256"]) and type(receipt["artifact_bytes"]) is int and receipt["artifact_bytes"] > 0, "phase artifact receipt drift")
            artifact_paths.add(relative); artifact = (root.parent / relative).resolve()
            req(root.parent.resolve() in artifact.parents and artifact.is_file(), "phase artifact missing on finalizer reread")
            payload = artifact.read_bytes(); req(len(payload) == receipt["artifact_bytes"] and hashlib.sha256(payload).hexdigest() == receipt["artifact_sha256"], "phase artifact finalizer bytes/hash drift")
            wrapper = json.loads(payload); disk = wrapper.get("gdn_phase_witness"); binding_row = wrapper.get("binding")
            req(set(wrapper) == {"schema_version", "binding", "gdn_phase_witness", "kv_ownership_witness"} and type(binding_row) is dict and binding_row.get("rank") == rank and binding_row.get("resident_count") == 8 and binding_row.get("kv_policy") == pre["selected_cell"]["kv_policy"] and binding_row.get("gdn_base_policy") == pre["selected_cell"]["gdn_base_policy"] and binding_row.get("phase") == receipt["phase"] and disk.get("phase") == receipt["phase"] and f"rank-{rank}-N-8-" in binding_row.get("cell_id", "") and binding_row["cell_id"].endswith("-ownership-witness"), "wrapper/GDN rank-cell-phase binding drift")
            req(hashlib.sha256(json.dumps(disk, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == receipt["gdn_sha256"], "phase artifact finalizer GDN hash drift")
        expected_calls = pre["acceptance"]["primary_memory_calls_per_rank"]
        req(type(zero["rank"]) is int and zero["rank"] == rank and all(type(zero[key]) is int for key in ("selected_builds", "selected_phases", "expected_primary_memory_calls", "primary_memory_calls_observed", "primary_memory_hook_events")) and zero["selected_builds"] == 1 and zero["selected_phases"] == 3 and zero["expected_primary_memory_calls"] == zero["primary_memory_calls_observed"] == expected_calls and zero["primary_call_coverage_proof"] is True and zero["primary_absence_proof"] is True and zero["primary_memory_hook_events"] == 0, "primary absence proof drift")
        primary.append(0); results.append({"rank": rank, "binding_sha256": sha(binding), "absence_sha256": sha(absence), "selected_rows": 18, "storage_rows": 1620, "borrowed_setup_aliases": 480, "setup_clone_edges": 0, "functional_rebind_calls": calls, "functional_rebind_edges": edges, "phase_artifacts": 3, "primary_calls": expected_calls})
    output = {"schema_version": "forkaudit-r40-v32-borrowed-transition-aggregate-v1", "rank_count": len(results), "rank_results": results, "total_selected_rows": sum(row["selected_rows"] for row in results), "total_storage_rows": sum(row["storage_rows"] for row in results), "total_borrowed_setup_aliases": sum(row["borrowed_setup_aliases"] for row in results), "total_setup_clone_edges": sum(row["setup_clone_edges"] for row in results), "total_functional_rebind_calls": sum(row["functional_rebind_calls"] for row in results), "total_functional_rebind_edges": sum(row["functional_rebind_edges"] for row in results), "total_phase_artifacts": sum(row["phase_artifacts"] for row in results), "total_primary_calls_observed": sum(row["primary_calls"] for row in results), "primary_events_by_rank": primary, "global_primary_memory_hook_events": sum(primary), "execution_bindings": bindings, "requires_cuda_smoke_before_science": True, "formal_gpu_execution": "clean-only-not-fault-campaign"}
    req(output["global_primary_memory_hook_events"] == 0, "global primary events")
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--terminal-root", type=Path, required=True); parser.add_argument("--capture-root", type=Path, required=True); parser.add_argument("--preregistration", type=Path, required=True); parser.add_argument("--expected-prereg-sha256", required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    terminal_root = validate_root(args.terminal_root); output, _ = ensure_output_absent(terminal_root, args.output)
    req(sha(args.preregistration) == args.expected_prereg_sha256, "prereg hash")
    publish_json_exclusive(terminal_root, output, finalize(args.capture_root, json.loads(args.preregistration.read_text())), expected_fields=AGGREGATE_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
