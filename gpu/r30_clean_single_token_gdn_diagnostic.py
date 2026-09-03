from __future__ import annotations

"""Post-discovery diagnostic for the invalid R29 one-token clean lane.

This is development evidence, not a held-out scientific result.  It rebuilds
one fresh clean N=2 case, executes the frozen one-token boundary action, and
records normalized storage identities and exact byte ranges for every GDN
state tensor.  Absolute pointers and Python object ids never leave the
process.  The diagnostic distinguishes a merely strict rebind predicate from
an actual request/base/peer storage overlap and shared-state mutation.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

import qcomem_forkaudit_storage_witness as storage_witness
import r29_execute_heldout_faults as executor


SCHEMA_VERSION = "forkaudit-r30-clean-single-token-gdn-diagnostic-v1"
STATE_FAMILIES = ("conv_states", "recurrent_states")
OWNER_NAMES = ("persistent", "request_0", "request_1")


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the minimal addressed byte interval for a strided dense view."""

    shape = tuple(int(value) for value in tensor.shape)
    stride = tuple(int(value) for value in tensor.stride())
    require(bool(shape) and len(shape) == len(stride), "state tensor rank drift")
    require(all(size > 0 for size in shape), "state tensor has an empty dimension")
    minimum = int(tensor.storage_offset())
    maximum = minimum
    for size, step in zip(shape, stride):
        displacement = (size - 1) * step
        minimum += min(displacement, 0)
        maximum += max(displacement, 0)
    require(minimum >= 0 and maximum >= minimum, "state tensor byte interval drift")
    element_size = int(tensor.element_size())
    return minimum * element_size, (maximum + 1) * element_size


@dataclass
class IdentityRegistry:
    storage_labels: dict[tuple[str, int, int], str] = field(default_factory=dict)
    object_labels: dict[int, str] = field(default_factory=dict)

    def storage_label(self, tensor: torch.Tensor) -> str:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
        if key not in self.storage_labels:
            self.storage_labels[key] = f"storage-{len(self.storage_labels):04d}"
        return self.storage_labels[key]

    def object_label(self, tensor: torch.Tensor) -> str:
        key = id(tensor)
        if key not in self.object_labels:
            self.object_labels[key] = f"tensor-{len(self.object_labels):04d}"
        return self.object_labels[key]


def _tensor_descriptor(
    tensor: torch.Tensor,
    *,
    registry: IdentityRegistry,
    owner: str,
    layer_index: int,
    family: str,
    state_index: int,
) -> dict[str, Any]:
    require(isinstance(tensor, torch.Tensor), "state value is not a tensor")
    byte_start, byte_end = byte_interval(tensor)
    element_size = int(tensor.element_size())
    tensor_nbytes = int(tensor.numel()) * element_size
    storage_nbytes = int(tensor.untyped_storage().nbytes())
    require(0 <= byte_start < byte_end <= storage_nbytes, "tensor byte range is invalid")
    return {
        "owner": owner,
        "layer_index": int(layer_index),
        "state_family": family,
        "state_index": int(state_index),
        "coordinate": f"layer:{int(layer_index)}/{family}/state:{int(state_index)}",
        "tensor_id": registry.object_label(tensor),
        "storage_id": registry.storage_label(tensor),
        "storage_nbytes": storage_nbytes,
        "byte_start": byte_start,
        "byte_end_exclusive": byte_end,
        "tensor_nbytes": tensor_nbytes,
        "shape": [int(value) for value in tensor.shape],
        "stride": [int(value) for value in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "dense_contiguous": bool(tensor.is_contiguous()),
        "content_sha256": executor.tensor_sha(tensor),
        "contains_absolute_pointer": False,
        "contains_python_object_id": False,
    }


def capture_snapshot(
    persistent: Any,
    group: Any,
    layer_indices: Sequence[int],
    registry: IdentityRegistry,
) -> dict[str, Any]:
    owners = {
        "persistent": persistent,
        "request_0": group.requests[0],
        "request_1": group.requests[1],
    }
    rows: list[dict[str, Any]] = []
    for owner in OWNER_NAMES:
        cache = owners[owner]
        for layer_index in layer_indices:
            layer = cache.layers[int(layer_index)]
            for family in STATE_FAMILIES:
                states = getattr(layer, family)
                require(isinstance(states, dict) and sorted(states) == [0], "state-index schema drift")
                rows.append(
                    _tensor_descriptor(
                        states[0],
                        registry=registry,
                        owner=owner,
                        layer_index=int(layer_index),
                        family=family,
                        state_index=0,
                    )
                )
    expected = len(OWNER_NAMES) * len(tuple(layer_indices)) * len(STATE_FAMILIES)
    require(len(rows) == expected, "snapshot tensor count drift")
    return {
        "row_count": len(rows),
        "rows": rows,
        "rows_sha256": executor.sha256_json(rows),
        "absolute_pointers_persisted": False,
        "python_object_ids_persisted": False,
    }


def _row_map(snapshot: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = snapshot["rows"]
    return {(row["owner"], row["coordinate"]): row for row in rows}


def relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    same_storage = left["storage_id"] == right["storage_id"]
    ranges_overlap = same_storage and max(left["byte_start"], right["byte_start"]) < min(
        left["byte_end_exclusive"], right["byte_end_exclusive"]
    )
    exact_range = (
        same_storage
        and left["byte_start"] == right["byte_start"]
        and left["byte_end_exclusive"] == right["byte_end_exclusive"]
    )
    return {
        "same_tensor_object": left["tensor_id"] == right["tensor_id"],
        "same_storage": same_storage,
        "ranges_overlap": ranges_overlap,
        "exact_byte_range_alias": exact_range,
        "storage_disjoint": not ranges_overlap,
        "content_equal": left["content_sha256"] == right["content_sha256"],
    }


def ownership_relations(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    rows = _row_map(snapshot)
    coordinates = sorted(
        coordinate for owner, coordinate in rows if owner == "persistent"
    )
    output: list[dict[str, Any]] = []
    for coordinate in coordinates:
        base = rows[("persistent", coordinate)]
        request_0 = rows[("request_0", coordinate)]
        request_1 = rows[("request_1", coordinate)]
        output.append(
            {
                "coordinate": coordinate,
                "state_family": base["state_family"],
                "request_0_vs_base": relation(request_0, base),
                "request_0_vs_peer": relation(request_0, request_1),
                "base_vs_peer": relation(base, request_1),
            }
        )
    return {
        "row_count": len(output),
        "rows": output,
        "rows_sha256": executor.sha256_json(output),
    }


def transition_relations(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    left = _row_map(before)
    right = _row_map(after)
    require(set(left) == set(right), "before/after coordinate drift")
    rows: list[dict[str, Any]] = []
    for owner, coordinate in sorted(left):
        pre = left[(owner, coordinate)]
        post = right[(owner, coordinate)]
        rows.append(
            {
                "owner": owner,
                "coordinate": coordinate,
                "state_family": pre["state_family"],
                "binding_changed": (
                    pre["tensor_id"] != post["tensor_id"]
                    or pre["storage_id"] != post["storage_id"]
                    or pre["byte_start"] != post["byte_start"]
                    or pre["byte_end_exclusive"] != post["byte_end_exclusive"]
                ),
                "content_changed": pre["content_sha256"] != post["content_sha256"],
                "pre_tensor_id": pre["tensor_id"],
                "post_tensor_id": post["tensor_id"],
                "pre_storage_id": pre["storage_id"],
                "post_storage_id": post["storage_id"],
                "pre_byte_range": [pre["byte_start"], pre["byte_end_exclusive"]],
                "post_byte_range": [post["byte_start"], post["byte_end_exclusive"]],
                "pre_content_sha256": pre["content_sha256"],
                "post_content_sha256": post["content_sha256"],
            }
        )
    return {
        "row_count": len(rows),
        "rows": rows,
        "rows_sha256": executor.sha256_json(rows),
    }


def _counts_by_family(rows: Sequence[Mapping[str, Any]], field_path: Sequence[str]) -> dict[str, int]:
    counts = {family: 0 for family in STATE_FAMILIES}
    for row in rows:
        value: Any = row
        for key in field_path:
            value = value[key]
        if value is True:
            counts[row["state_family"]] += 1
    return counts


def summarize(
    before_relations: Mapping[str, Any],
    after_relations: Mapping[str, Any],
    transitions: Mapping[str, Any],
) -> dict[str, Any]:
    pre_rows = before_relations["rows"]
    post_rows = after_relations["rows"]
    transition_rows = transitions["rows"]
    summary = {
        "before_request_0_vs_base_exact_alias": _counts_by_family(
            pre_rows, ("request_0_vs_base", "exact_byte_range_alias")
        ),
        "before_request_0_vs_peer_exact_alias": _counts_by_family(
            pre_rows, ("request_0_vs_peer", "exact_byte_range_alias")
        ),
        "after_request_0_vs_base_overlap": _counts_by_family(
            post_rows, ("request_0_vs_base", "ranges_overlap")
        ),
        "after_request_0_vs_base_disjoint": _counts_by_family(
            post_rows, ("request_0_vs_base", "storage_disjoint")
        ),
        "after_request_0_vs_peer_overlap": _counts_by_family(
            post_rows, ("request_0_vs_peer", "ranges_overlap")
        ),
        "after_request_0_vs_peer_disjoint": _counts_by_family(
            post_rows, ("request_0_vs_peer", "storage_disjoint")
        ),
    }
    for owner in OWNER_NAMES:
        owner_rows = [row for row in transition_rows if row["owner"] == owner]
        summary[f"{owner}_binding_changed"] = _counts_by_family(
            owner_rows, ("binding_changed",)
        )
        summary[f"{owner}_content_changed"] = _counts_by_family(
            owner_rows, ("content_changed",)
        )
    conv_overlap = summary["after_request_0_vs_base_overlap"]["conv_states"]
    peer_conv_overlap = summary["after_request_0_vs_peer_overlap"]["conv_states"]
    base_conv_changed = summary["persistent_content_changed"]["conv_states"]
    peer_conv_changed = summary["request_1_content_changed"]["conv_states"]
    request_conv_rebound = summary["request_0_binding_changed"]["conv_states"]
    summary["diagnosis"] = {
        "rebind_predicate_merely_overstrict": False,
        "actual_cross_owner_alias_persists_after_single_token": conv_overlap > 0
        and peer_conv_overlap > 0,
        "shared_base_and_peer_content_mutated": base_conv_changed > 0
        and peer_conv_changed > 0,
        "request_conv_binding_rebound": request_conv_rebound > 0,
        "classification": "single-token-conv-state-ownership-bug",
        "scientific_use": "development_only_not_heldout",
    }
    return summary


def exception_receipt(operation: Any) -> dict[str, Any]:
    try:
        operation()
    except BaseException as exc:
        receipt = executor._exception_record(exc)
        executor._clear_exception(exc)
        return {"raised": True, "exception": receipt}
    return {"raised": False, "exception": None}


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(sha256_file(Path(__file__).resolve()) == args.expected_script_sha256, "diagnostic source SHA drift")
    execution_raw = executor.read_bound_file(
        args.execution_input,
        args.expected_execution_input_sha256,
        "frozen R29 execution input",
    )
    execution_input = executor.validate_execution_input(json.loads(execution_raw))
    require(
        execution_input["code"]["executor_sha256"] == args.expected_executor_sha256,
        "executor binding argument drift",
    )
    executor_path = Path(execution_input["code"]["executor_path"])
    require(sha256_file(executor_path) == args.expected_executor_sha256, "executor source SHA drift")
    require(not args.output.exists(), "diagnostic output already exists")

    runtime_args = SimpleNamespace(rank=0, expected_gpu_uuid=args.expected_gpu_uuid)
    runtime = executor._load_runtime(runtime_args, execution_input)
    registry = IdentityRegistry()
    backend = ""
    persistent = group = persistent_guard = request_guard = kv_guard = None
    source_guard = None
    logits = None
    result: dict[str, Any] | None = None
    after_allocator: dict[str, int] | None = None
    with torch.inference_mode():
        warmup = executor._discarded_warmup(runtime)
        require(runtime.allocator_baseline is not None, "allocator baseline missing")
        before_allocator = executor._snapshot_allocator()
        require(before_allocator == runtime.allocator_baseline, "allocator not at baseline")
        try:
            (
                persistent,
                group,
                persistent_guard,
                request_guard,
                kv_guard,
                source_guard,
            ) = executor._build_fresh_case(runtime)
            before = capture_snapshot(
                persistent, group, runtime.plan.linear_layer_indices, registry
            )
            before_relations = ownership_relations(before)
            ledger, backend = executor._make_backend(runtime, group, 1)
            logits, model_step = executor._model_step(runtime, group, backend)
            kernel_ledger = executor.rr2._pointer_free_kernel_ledger(
                ledger.verify_complete()
            )
            after = capture_snapshot(
                persistent, group, runtime.plan.linear_layer_indices, registry
            )
            after_relations = ownership_relations(after)
            transitions = transition_relations(before, after)
            request_guard_result = exception_receipt(
                lambda: storage_witness.verify_request_gdn_binding_guard(
                    request_guard,
                    group.requests,
                    completed_request_indices=(0,),
                )
            )
            persistent_guard_result = exception_receipt(
                lambda: storage_witness.verify_persistent_gdn_guard(
                    persistent_guard, persistent
                )
            )
            source_after = executor.resident.source_document_physical_digests(
                persistent, runtime.plan.full_attention_layer_indices
            )
            require(source_after == source_guard, "persistent KV changed")
            summary = summarize(before_relations, after_relations, transitions)
            result = {
                "schema_version": SCHEMA_VERSION,
                "status": "completed_post_discovery_development_diagnostic",
                "scientific_valid": False,
                "heldout_claim_allowed": False,
                "paper_import_allowed": False,
                "purpose": "diagnose the operational-invalid R29 attempt-C clean lane",
                "source_bindings": {
                    "diagnostic_sha256": args.expected_script_sha256,
                    "r29_executor_sha256": args.expected_executor_sha256,
                    "r29_execution_input_raw_sha256": args.expected_execution_input_sha256,
                    "imported_rr2_code_ledger_raw_sha256": execution_input["code"][
                        "imported_rr2_code_ledger_raw_sha256"
                    ],
                },
                "hardware": runtime.hardware,
                "input_receipt": runtime.input_receipt,
                "discarded_warmup": warmup,
                "action": {
                    "request_index": 0,
                    "input_coordinate": "frozen_query_bank[rank][0][31]",
                    "token_count": 1,
                    "model_step": model_step,
                    "kernel_ledger": kernel_ledger,
                    "full_logit_sha256": executor.tensor_sha(logits),
                },
                "before": before,
                "after": after,
                "before_ownership_relations": before_relations,
                "after_ownership_relations": after_relations,
                "transition_relations": transitions,
                "registered_guard_results": {
                    "request_binding_guard": request_guard_result,
                    "persistent_gdn_guard": persistent_guard_result,
                },
                "summary": summary,
                "absolute_pointers_persisted": False,
                "python_object_ids_persisted": False,
            }
        finally:
            if backend:
                executor.rr2._unregister_backends([backend])
            backend = ""
            logits = None
            persistent = group = persistent_guard = request_guard = kv_guard = None
            source_guard = None
            after_allocator = executor._cleanup_allocator()
            require(
                after_allocator == runtime.allocator_baseline,
                "diagnostic allocator baseline did not restore",
            )
    require(result is not None, "diagnostic result was not constructed")
    require(after_allocator is not None, "diagnostic cleanup receipt missing")
    result["cleanup"] = {
        "allocator_before": before_allocator,
        "allocator_after": after_allocator,
        "allocator_baseline": dict(runtime.allocator_baseline),
        "allocator_baseline_exact": after_allocator == runtime.allocator_baseline,
        "registered_backend_restored": True,
        "fresh_case_disposed": True,
    }
    executor.write_json_atomic(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--execution-input", type=Path, required=True)
    value.add_argument("--expected-execution-input-sha256", required=True)
    value.add_argument("--expected-executor-sha256", required=True)
    value.add_argument("--expected-script-sha256", required=True)
    value.add_argument("--expected-gpu-uuid", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    run(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
