from __future__ import annotations

import contextvars
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from r40_real_binding import ActualBindingVerifier, require


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


def install_real_binding_hook(runner_module: Any, preregistration: dict[str, Any], *, capture_root: Path, rank: int):
    original_witness = runner_module._run_ownership_witness_cell
    original_build = runner_module.build_resident_request_group
    original_phase = runner_module._write_witness_phase
    active = contextvars.ContextVar("r40_v4_real_binding", default=None)
    selected_cell = preregistration["selected_cell"]

    def witness(*args: Any, **kwargs: Any):
        metadata = {"resident_count": int(kwargs["resident_count"]), "kv_policy": kwargs["kv_policy"], "gdn_base_policy": kwargs["gdn_base_policy"], "cell_role": "ownership_witness"}
        if any(metadata[key] != selected_cell[key] for key in selected_cell):
            return original_witness(*args, **kwargs)
        state = {"verifier": None, "receipts": []}
        token = active.set(state)
        try:
            result = original_witness(*args, **kwargs)
            require(state["verifier"] is not None, "real builder hook did not run")
            require([row["phase"] for row in state["receipts"]] == ["setup_pre_transition", "post_transition", "post_generation"], "real phase coverage drift")
            payload = _seal({
                "schema_version": "forkaudit-r40-v4-real-binding-rank-v1",
                "experiment_id": preregistration["experiment_id"],
                "rank": int(rank), "selected_cell": metadata,
                "phase_order": [row["phase"] for row in state["receipts"]],
                "phase_receipts": state["receipts"],
                "source_reference_coordinate_count": len(state["verifier"].source),
                "actual_selected_rows_verified": sum(row["selected_rows_verified"] for row in state["receipts"]),
                "real_builder_verified": True, "actual_phase_serializer_verified": True,
                "off_path_candidate_detector_used": False,
                "primary_memory_hook_events": 0,
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
        verifier = ActualBindingVerifier(cache, preregistration["selected_coordinates"])
        state["verifier"] = verifier
        group = original_build(cache, plan, **kwargs)
        verifier.verify_built_group(group)
        return group

    def phase(*args: Any, **kwargs: Any):
        result = original_phase(*args, **kwargs)
        state = active.get()
        if state is not None:
            require(isinstance(result, tuple) and len(result) == 2, "phase return schema drift")
            receipt = state["verifier"].verify_serialized_phase(result[1], str(kwargs["phase"]))
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
