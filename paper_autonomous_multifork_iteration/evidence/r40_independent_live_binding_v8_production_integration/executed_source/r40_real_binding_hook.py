from __future__ import annotations

import contextvars
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from r40_real_binding import ActualBindingVerifier, require
from r40_clone_lineage import CloneLineageMode

GLOBAL_HOOK_COUNTERS = {"selected_builds": 0, "selected_phases": 0, "primary_memory_events": 0}


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["payload_sha256"] = None
    payload["payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def _write_new(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), "refusing to overwrite real-binding artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    os.link(temporary, path)
    temporary.unlink()


def install_real_binding_hook(runner_module: Any, preregistration: dict[str, Any], *, capture_root: Path, rank: int, execution_bindings: dict[str, str]):
    original_witness = runner_module._run_ownership_witness_cell
    original_build = runner_module.build_resident_request_group
    original_phase = runner_module._write_witness_phase
    active = contextvars.ContextVar("r40_v4_real_binding", default=None)
    selected_cell = preregistration["selected_cell"]

    def witness(*args: Any, **kwargs: Any):
        metadata = {"resident_count": int(kwargs["resident_count"]), "kv_policy": kwargs["kv_policy"], "gdn_base_policy": kwargs["gdn_base_policy"], "cell_role": "ownership_witness"}
        if any(metadata[key] != selected_cell[key] for key in selected_cell):
            return original_witness(*args, **kwargs)
        state = {"verifier": None, "receipts": [], "build_count": 0, "metadata": metadata}
        token = active.set(state)
        try:
            result = original_witness(*args, **kwargs)
            require(state["verifier"] is not None, "real builder hook did not run")
            require(state["build_count"] == 1, "selected real builder count drift")
            require([row["phase"] for row in state["receipts"]] == ["setup_pre_transition", "post_transition", "post_generation"], "real phase coverage drift")
            payload = _seal({
                "schema_version": "forkaudit-r40-v4-real-binding-rank-v1",
                "experiment_id": preregistration["experiment_id"],
                "rank": int(rank), "selected_cell": metadata,
                "phase_order": [row["phase"] for row in state["receipts"]],
                "phase_receipts": state["receipts"],
                "source_reference_coordinate_count": len(state["verifier"].source),
                "actual_selected_rows_verified": sum(row["selected_rows_verified"] for row in state["receipts"]),
                "actual_storage_rows_verified": sum(row["actual_storage_rows_verified"] for row in state["receipts"]),
                "count_vector": {
                    "source_reference_coordinates": len(state["verifier"].source),
                    "selected_rows_by_phase": [row["selected_rows_verified"] for row in state["receipts"]],
                    "storage_rows_by_phase": [row["actual_storage_rows_verified"] for row in state["receipts"]],
                    "primary_memory_hook_events": 0
                },
                "real_builder_verified": True, "actual_phase_serializer_verified": True,
                "off_path_candidate_detector_used": False,
                "producer_coverage": {"prebuild_reference_frozen": True, "real_group_observed": True, "actual_serializer_rows_observed": True, "persistent_rechecked_each_phase": True, "all_storage_rows_normalized_against_live_keys": True},
                "primary_memory_hook_events": 0,
                "global_hook_counters": dict(GLOBAL_HOOK_COUNTERS),
                "execution_bindings": dict(execution_bindings),
                "formal_gpu_execution": "result-only-if-written-by-authorized-launch",
                "payload_sha256": None,
            })
            _write_new(capture_root / f"rank-{rank}/raw/real-binding.json", payload)
            return result
        finally:
            active.reset(token)

    def build(cache: Any, plan: Any, **kwargs: Any):
        state = active.get()
        if state is None:
            return original_build(cache, plan, **kwargs)
        # Reference is frozen before the unchanged real builder executes.
        verifier = ActualBindingVerifier(cache, preregistration["selected_coordinates"], plan.linear_layer_indices)
        state["verifier"] = verifier
        require(int(kwargs["resident_count"]) == int(state.get("metadata", {}).get("resident_count", kwargs["resident_count"])), "build N/cell drift")
        require(kwargs["policy"] == preregistration["selected_cell"]["kv_policy"] and kwargs["gdn_base_policy"] == preregistration["selected_cell"]["gdn_base_policy"], "build policy/cell drift")
        lineage = CloneLineageMode(verifier)
        with lineage:
            group = original_build(cache, plan, **kwargs)
        state["build_count"] += 1; GLOBAL_HOOK_COUNTERS["selected_builds"] += 1
        require(int(group.resident_count) == int(kwargs["resident_count"]) and len(group.requests) == int(kwargs["resident_count"]), "group N metadata drift")
        require(group.policy == kwargs["policy"] and group.audit["gdn_base_policy"] == kwargs["gdn_base_policy"], "group policy metadata drift")
        verifier.verify_built_group(group)
        verifier.attach_lineage_receipt(lineage.bind_returned_group(group, preregistration["selected_coordinates"]))
        return group

    def phase(*args: Any, **kwargs: Any):
        result = original_phase(*args, **kwargs)
        state = active.get()
        if state is not None:
            require(isinstance(result, tuple) and len(result) == 2, "phase return schema drift")
            receipt = state["verifier"].verify_serialized_phase(result[1], str(kwargs["phase"]), kwargs["completed_request_indices"])
            GLOBAL_HOOK_COUNTERS["selected_phases"] += 1
            state["receipts"].append(receipt)
        return result

    runner_module._run_ownership_witness_cell = witness
    runner_module.build_resident_request_group = build
    runner_module._write_witness_phase = phase

    def restore():
        runner_module._run_ownership_witness_cell = original_witness
        runner_module.build_resident_request_group = original_build
        runner_module._write_witness_phase = original_phase
    return restore


__all__ = ["install_real_binding_hook"]
