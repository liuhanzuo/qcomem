from __future__ import annotations

import contextvars
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from r40_real_binding import ActualBindingVerifier, require
from r40_passive_clone_lineage import PassiveCloneLineageMode, PersistentSourceRegistry, storage_descriptor
from r40_real_binding import digest, storage_key, tensor_at

GLOBAL_HOOK_COUNTERS = {"selected_builds": 0, "selected_phases": 0, "primary_memory_calls_observed": 0, "primary_memory_hook_events": 0}


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

def _bind_phase_artifact(reference: dict[str, Any], artifact_root: Path, returned_gdn: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    require(set(reference) == {"relative_path","sha256","bytes"}, "phase artifact reference schema drift")
    root=artifact_root.resolve(); path=(root/reference["relative_path"]).resolve()
    require(root in path.parents and path.is_file(), "phase artifact path drift")
    payload=path.read_bytes(); require(len(payload)==reference["bytes"] and hashlib.sha256(payload).hexdigest()==reference["sha256"], "phase artifact bytes/hash drift")
    wrapper=json.loads(payload); disk_gdn=wrapper.get("gdn_phase_witness")
    require(hashlib.sha256(json.dumps(disk_gdn,sort_keys=True,separators=(",", ":")).encode()).hexdigest()==hashlib.sha256(json.dumps(returned_gdn,sort_keys=True,separators=(",", ":")).encode()).hexdigest(), "disk/returned GDN drift")
    return path,{"artifact_relative_path":reference["relative_path"],"artifact_sha256":reference["sha256"],"artifact_bytes":reference["bytes"],"gdn_sha256":hashlib.sha256(json.dumps(returned_gdn,sort_keys=True,separators=(",", ":")).encode()).hexdigest()}


def _published_phase_relative_path(*, reference: dict[str, Any], artifact_root: Path, capture_root: Path, rank: int) -> str:
    temporary_root = artifact_root.resolve()
    result_root = capture_root.resolve().parent
    published_root = (result_root / "primary/raw").resolve()
    require(temporary_root.parent == published_root, "phase artifact temporary root parent drift")
    require(temporary_root.name.startswith(f".forkaudit-rank-{rank}-") and len(temporary_root.name) > len(f".forkaudit-rank-{rank}-"), "phase artifact temporary root identity drift")
    relative_text = reference.get("relative_path")
    require(isinstance(relative_text, str) and relative_text != "", "phase artifact relative path type drift")
    relative = PurePosixPath(relative_text)
    require(not relative.is_absolute() and relative.as_posix() == relative_text and relative.parts[0] == f"rank-{rank}" and all(part not in ("", ".", "..") for part in relative.parts), "phase artifact relative rank/path drift")
    published = (published_root / Path(*relative.parts)).resolve()
    require(published_root in published.parents and temporary_root not in published.parents and not published.exists(), "phase artifact published path containment/freshness drift")
    return published.relative_to(result_root).as_posix()


def install_real_binding_hook(runner_module: Any, preregistration: dict[str, Any], *, capture_root: Path, rank: int, execution_bindings: dict[str, str]):
    for key in GLOBAL_HOOK_COUNTERS: GLOBAL_HOOK_COUNTERS[key]=0
    require(set(execution_bindings)==set(preregistration["required_execution_binding_fields"]),"execution binding field-set drift")
    require(all(isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value) for value in execution_bindings.values()),"execution binding SHA drift")
    original_witness = runner_module._run_ownership_witness_cell
    original_build = runner_module.build_resident_request_group
    original_phase = runner_module._write_witness_phase
    require(hasattr(runner_module,"_run_clean_memory_cell") and hasattr(runner_module,"_run_formal_factorial_cells"),"hash-bound runner clean-memory/factorial interface missing")
    original_memory = runner_module._run_clean_memory_cell
    original_factorial = runner_module._run_formal_factorial_cells
    require(hasattr(runner_module,"_gpu_round_robin_generate"),"hash-bound runner generation interface missing")
    original_generation = runner_module._gpu_round_robin_generate
    active = contextvars.ContextVar("forkaudit_r40_v30_real_binding_context", default=None)
    formal_scope = contextvars.ContextVar("forkaudit_r40_v30_formal_factorial_scope", default=False)
    selected_cell = preregistration["selected_cell"]

    def witness(*args: Any, **kwargs: Any):
        metadata = {"resident_count": int(kwargs["resident_count"]), "kv_policy": kwargs["kv_policy"], "gdn_base_policy": kwargs["gdn_base_policy"], "cell_role": "ownership_witness"}
        if any(metadata[key] != selected_cell[key] for key in selected_cell):
            return original_witness(*args, **kwargs)
        state = {"verifier": None, "receipts": [], "build_count": 0, "metadata": metadata, "lineage_summary": None}
        token = active.set(state)
        try:
            result = original_witness(*args, **kwargs)
            require(state["verifier"] is not None, "real builder hook did not run")
            require(state["build_count"] == 1, "selected real builder count drift")
            require([row["phase"] for row in state["receipts"]] == ["setup_pre_transition", "post_transition", "post_generation"], "real phase coverage drift")
            functional_ledger = state["verifier"].functional_rebind_receipt()
            require(functional_ledger["call_count"] == 64 and functional_ledger["edge_count"] == 3840 and functional_ledger["edges_per_call"] == 60, "functional rebind ledger cardinality drift")
            require(all(functional_ledger[key] is True for key in ("all_new_tensor_objects","all_new_storages","all_descriptors_authorized","all_contents_recorded")), "functional rebind ledger predicate drift")
            payload = _seal({
                "schema_version": "forkaudit-r40-v30-borrowed-transition-rank-v1",
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
                    "full_live_rows_by_phase": [row["full_live_rows_verified"] for row in state["receipts"]],
                    "generation_calls_by_phase": [row["generation_calls_verified"] for row in state["receipts"]],
                    "functional_rebind_edges_by_phase": [row["functional_rebind_edges_verified"] for row in state["receipts"]],
                    "request_rebind_counts_by_phase": [row["request_rebind_counts"] for row in state["receipts"]],
                    "private_request_rows_by_phase": [row["private_request_rows_verified"] for row in state["receipts"]],
                    "borrowed_request_rows_by_phase": [row["borrowed_request_rows_verified"] for row in state["receipts"]],
                    "primary_memory_hook_events": GLOBAL_HOOK_COUNTERS["primary_memory_hook_events"]
                },
                "real_builder_verified": True, "actual_phase_serializer_verified": True,
                "off_path_candidate_detector_used": False,
                "producer_coverage": {"prebuild_reference_frozen": True, "real_group_observed": True, "borrowed_setup_exact_aliases_observed": True, "functional_rebind_endpoints_observed": True, "actual_serializer_rows_observed": True, "persistent_rechecked_each_phase": True, "all_storage_rows_normalized_against_live_keys": True},
                "primary_memory_hook_events": GLOBAL_HOOK_COUNTERS["primary_memory_hook_events"],
                "global_hook_counters": dict(GLOBAL_HOOK_COUNTERS),
                "execution_bindings": dict(execution_bindings),
                "lineage_summary": state["lineage_summary"],
                "lineage_summary_sha256": hashlib.sha256(json.dumps(state["lineage_summary"],sort_keys=True,separators=(",", ":")).encode()).hexdigest(),
                "lineage_receipt": state["verifier"].lineage_receipt,
                "functional_rebind_ledger": functional_ledger,
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
        verifier = ActualBindingVerifier(cache, preregistration["selected_coordinates"], plan.linear_layer_indices, setup_policy=kwargs["gdn_base_policy"])
        state["verifier"] = verifier
        require(int(kwargs["resident_count"]) == int(state.get("metadata", {}).get("resident_count", kwargs["resident_count"])), "build N/cell drift")
        require(kwargs["policy"] == preregistration["selected_cell"]["kv_policy"] and kwargs["gdn_base_policy"] == preregistration["selected_cell"]["gdn_base_policy"], "build policy/cell drift")
        all_coordinates=[(int(index),family,0) for index in plan.linear_layer_indices for family in ("conv","recurrent")]
        registry=PersistentSourceRegistry(cache,all_coordinates)
        lineage = PassiveCloneLineageMode(registry)
        with lineage:
            group = original_build(cache, plan, **kwargs)
        state["build_count"] += 1; GLOBAL_HOOK_COUNTERS["selected_builds"] += 1
        require(int(group.resident_count) == int(kwargs["resident_count"]) and len(group.requests) == int(kwargs["resident_count"]), "group N metadata drift")
        require(group.policy == kwargs["policy"] and group.audit["gdn_base_policy"] == kwargs["gdn_base_policy"], "group policy metadata drift")
        verifier.verify_built_group(group)
        if kwargs["gdn_base_policy"] == "materialize-request-base-functional-rebind":
            state["lineage_summary"] = lineage.verify_materialized(group.requests, all_coordinates, require_direct_clone=True)
        else:
            state["lineage_summary"] = lineage.verify_borrowed(group.requests, all_coordinates)
        verifier.attach_lineage_capability(lineage.issue_capability(group.requests,preregistration["selected_coordinates"],verifier,setup_policy=kwargs["gdn_base_policy"]))
        return group

    def phase(*args: Any, **kwargs: Any):
        result = original_phase(*args, **kwargs)
        state = active.get()
        if state is not None:
            require(state["verifier"].lineage_receipt is not None and state["verifier"].lineage_receipt.get("opaque_capability_consumed") is True, "lineage capability gate unsatisfied")
            require(isinstance(result, tuple) and len(result) == 2, "phase return schema drift")
            artifact_path=(Path(kwargs["artifact_root"]).resolve()/str(result[0].get("relative_path","__invalid__"))).resolve() if isinstance(result[0],dict) else None
            try:
                artifact_path,artifact_receipt=_bind_phase_artifact(result[0],Path(kwargs["artifact_root"]),result[1])
                artifact_receipt["artifact_relative_path"]=_published_phase_relative_path(reference=result[0],artifact_root=Path(kwargs["artifact_root"]),capture_root=capture_root,rank=rank)
                receipt = state["verifier"].verify_serialized_phase(result[1], str(kwargs["phase"]), kwargs["completed_request_indices"])
                receipt.update(artifact_receipt); GLOBAL_HOOK_COUNTERS["selected_phases"] += 1; state["receipts"].append(receipt)
            except BaseException:
                if artifact_path is not None and artifact_path.exists(): artifact_path.unlink()
                raise
        return result

    runner_module._run_ownership_witness_cell = witness
    runner_module.build_resident_request_group = build
    runner_module._write_witness_phase = phase
    def factorial(*args:Any,**kwargs:Any):
        token=formal_scope.set(True)
        try:return original_factorial(*args,**kwargs)
        finally:formal_scope.reset(token)

    def memory(*args:Any,**kwargs:Any):
        if formal_scope.get():
            GLOBAL_HOOK_COUNTERS["primary_memory_calls_observed"] += 1
            if active.get() is not None:
                GLOBAL_HOOK_COUNTERS["primary_memory_hook_events"] += 1
                raise RuntimeError("primary clean-memory call occurred inside active ownership context")
        return original_memory(*args,**kwargs)
    runner_module._run_formal_factorial_cells=factorial
    runner_module._run_clean_memory_cell=memory
    def generation(*args:Any,**kwargs:Any):
        state=active.get()
        if state is None:return original_generation(*args,**kwargs)
        original_after=kwargs.get("after_step")
        def after(round_index:int,request_index:int):
            state["verifier"].observe_generation_step(round_index,request_index)
            if original_after is not None:original_after(round_index,request_index)
        kwargs=dict(kwargs);kwargs["after_step"]=after
        return original_generation(*args,**kwargs)
    runner_module._gpu_round_robin_generate=generation

    def restore():
        runner_module._run_ownership_witness_cell = original_witness
        runner_module.build_resident_request_group = original_build
        runner_module._write_witness_phase = original_phase
        runner_module._run_clean_memory_cell=original_memory
        runner_module._run_formal_factorial_cells=original_factorial
        runner_module._gpu_round_robin_generate=original_generation
    return restore


def global_absence_receipt() -> dict[str,int]: return dict(GLOBAL_HOOK_COUNTERS)
__all__ = ["install_real_binding_hook","global_absence_receipt"]
