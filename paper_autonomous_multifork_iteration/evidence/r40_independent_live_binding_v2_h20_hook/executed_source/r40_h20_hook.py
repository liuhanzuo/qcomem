from __future__ import annotations

"""Install the selected-cell Qwen/H20 pre-binder challenge without editing the runner."""

import contextvars
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from r40_h20_binding_protocol import (
    REGISTRATION_SCHEMA,
    coordinate_key,
    protocol_slot_id,
    require,
    seal,
    selected_slot_map,
)
from r40_h20_observer import ObserverClient, detect, observe
from r40_h20_registrar import RegistrarClient, RegistrationOracle


PHASE_SETUP = "setup_pre_transition"
PHASE_TRANSITION = "post_transition"


def _get_tensor(persistent: Any, group: Any, coordinate: Mapping[str, Any]) -> torch.Tensor:
    owner = (
        persistent
        if coordinate["owner_kind"] == "persistent"
        else group.requests[int(coordinate["request_index"])]
    )
    layer = owner.layers[int(coordinate["layer_index"])]
    states = (
        layer.conv_states
        if coordinate["state_family"] == "conv"
        else layer.recurrent_states
    )
    tensor = states[int(coordinate["state_index"])]
    require(isinstance(tensor, torch.Tensor), "candidate state is not a tensor")
    return tensor


def _candidate_items(
    preregistration: Mapping[str, Any], persistent: Any, group: Any
) -> list[dict[str, Any]]:
    rows = selected_slot_map(preregistration)
    return [
        {"slot_id": slot, "tensor": _get_tensor(persistent, group, coordinate)}
        for slot, coordinate in sorted(rows.items())
    ]


class _LocalRegistrar:
    def __init__(self, preregistration: Mapping[str, Any]) -> None:
        self.oracle = RegistrationOracle(preregistration)
        self.worker_pid = os.getpid()

    def register(self, event: Mapping[str, Any]) -> dict[str, Any]:
        return self.oracle.register_containers(event)

    def snapshot(self) -> dict[str, Any]:
        return self.oracle.snapshot()

    def close(self) -> None:
        return None


class _LocalObserver:
    def __init__(self, preregistration: Mapping[str, Any]) -> None:
        self.preregistration = preregistration
        self.worker_pid = os.getpid()
        self.count = 0

    def capture(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        self.count += 1
        return observe(items, self.preregistration)

    def close(self) -> int:
        return self.count


class BindingCampaignSession:
    def __init__(
        self,
        preregistration: Mapping[str, Any],
        *,
        rank: int,
        metadata: Mapping[str, Any],
        process_workers: bool,
    ) -> None:
        self.preregistration = dict(preregistration)
        self.rank = int(rank)
        self.metadata = dict(metadata)
        self.registrar = (
            RegistrarClient(self.preregistration)
            if process_workers
            else _LocalRegistrar(self.preregistration)
        )
        self.observer = (
            ObserverClient(self.preregistration)
            if process_workers
            else _LocalObserver(self.preregistration)
        )
        self.process_workers = bool(process_workers)
        self.persistent: Any = None
        self.group: Any = None
        self.plan: Any = None
        self.initial_handles: dict[str, torch.Tensor] = {}
        self.registration_receipts: list[dict[str, Any]] = []
        self.fault_results: list[dict[str, Any]] = []
        self.phase_order: list[str] = []
        self.closed = False

    def _send_all_owner_layers(self, operation: str, owners: list[tuple[str, int | None, Any]]) -> None:
        require(operation in {"initial", "refresh"}, "registration operation drift")
        for owner_kind, request_index, owner in owners:
            for layer_index in self.plan.linear_layer_indices:
                layer = owner.layers[int(layer_index)]
                event = {
                    "schema_version": REGISTRATION_SCHEMA,
                    "operation": operation,
                    "owner_kind": owner_kind,
                    "request_index": request_index,
                    "layer_index": int(layer_index),
                    "conv_states": layer.conv_states,
                    "recurrent_states": layer.recurrent_states,
                }
                self.registration_receipts.append(self.registrar.register(event))

    def register_initial(self, persistent: Any, group: Any, plan: Any) -> None:
        require(self.persistent is None, "initial registration repeated")
        self.persistent, self.group, self.plan = persistent, group, plan
        owners = [("persistent", None, persistent)] + [
            ("request", request_index, request)
            for request_index, request in enumerate(group.requests)
        ]
        # The producer hook emits every raw GDN layer container for every owner;
        # it does not use the selected-coordinate list to choose slot events.
        self._send_all_owner_layers("initial", owners)
        oracle = self.registrar.snapshot()
        require(oracle["row_count"] == 6, "initial registrar coverage drift")
        self.initial_handles = {
            item["slot_id"]: item["tensor"]
            for item in _candidate_items(self.preregistration, persistent, group)
        }

    def refresh_request_zero(self) -> None:
        require(self.group is not None and self.plan is not None, "refresh before initial")
        # Again emit all raw linear-layer containers for request zero.  The
        # registrar independently chooses and advances only selected slots.
        self._send_all_owner_layers(
            "refresh", [("request", 0, self.group.requests[0])]
        )

    def _fault_items(self, fault: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
        clean = {
            item["slot_id"]: item["tensor"]
            for item in _candidate_items(self.preregistration, self.persistent, self.group)
        }
        mutated = dict(clean)
        targets = [protocol_slot_id(row) for row in fault["targets"]]
        sources = [protocol_slot_id(row) for row in fault.get("sources", [])]
        kind = fault["kind"]
        if kind == "coherent_slot_swap":
            require(len(targets) == 2, "swap target drift")
            mutated[targets[0]], mutated[targets[1]] = clean[targets[1]], clean[targets[0]]
        elif kind == "stale_handle_after_rebind":
            require(len(targets) == 1, "stale target drift")
            mutated[targets[0]] = self.initial_handles[targets[0]]
            require(mutated[targets[0]] is not clean[targets[0]], "stale target did not rebind")
        elif kind in {"cross_layer_substitution", "request_base_role_misbinding"}:
            require(len(targets) == len(sources) == 1, "substitution target drift")
            mutated[targets[0]] = clean[sources[0]]
        else:
            raise RuntimeError(f"unknown fault kind: {kind}")
        changed = sorted(slot for slot in clean if mutated[slot] is not clean[slot])
        require(changed == sorted(targets), "fault changed unexpected live slots")
        return [
            {"slot_id": slot, "tensor": mutated[slot]} for slot in sorted(mutated)
        ], changed

    def run_phase(self, phase: str) -> None:
        require(phase not in self.phase_order, "campaign phase repeated")
        if phase == PHASE_TRANSITION:
            self.refresh_request_zero()
        self.phase_order.append(phase)
        faults = [row for row in self.preregistration["faults"] if row["phase"] == phase]
        require(faults, "campaign phase has no frozen faults")
        oracle = self.registrar.snapshot()
        for fault in faults:
            clean_items = _candidate_items(self.preregistration, self.persistent, self.group)
            clean_capture = self.observer.capture(clean_items)
            clean_detector = detect(oracle, clean_capture)
            require(clean_detector["passed"] is True, "matched clean failed")
            mutant_items, changed = self._fault_items(fault)
            mutant_capture = self.observer.capture(mutant_items)
            mutant_detector = detect(oracle, mutant_capture)
            require(mutant_detector["passed"] is False, "live-binding mutant escaped")
            require(
                all(code in mutant_detector["failure_codes"] for code in fault["required_detection_codes"]),
                "required live-binding detector code missing",
            )
            self.fault_results.append(
                {
                    "fault_id": fault["fault_id"],
                    "phase": phase,
                    "changed_slot_ids": changed,
                    "semantic_labels_mutated": False,
                    "clean_detector": clean_detector,
                    "mutant_detector": mutant_detector,
                    "registration_acknowledged_before_capture": True,
                    "oracle_payload_sha256": oracle["payload_sha256"],
                    "clean_payload_sha256": clean_capture["payload_sha256"],
                    "mutant_payload_sha256": mutant_capture["payload_sha256"],
                }
            )

    def payload(self) -> dict[str, Any]:
        require(
            {row["fault_id"] for row in self.fault_results}
            == {row["fault_id"] for row in self.preregistration["faults"]},
            "fault campaign incomplete",
        )
        require(self.phase_order == [PHASE_SETUP, PHASE_TRANSITION], "phase order drift")
        return seal(
            {
                "schema_version": "forkaudit-r40-h20-rank-live-binding-v1",
                "experiment_id": self.preregistration["experiment_id"],
                "rank": self.rank,
                "selected_cell": self.metadata,
                "process_workers": self.process_workers,
                "producer_pid": os.getpid(),
                "registrar_pid": self.registrar.worker_pid,
                "observer_pid": self.observer.worker_pid,
                "process_separated": len(
                    {os.getpid(), self.registrar.worker_pid, self.observer.worker_pid}
                )
                == 3,
                "producer_manifest_sent": False,
                "producer_slot_ids_sent_to_registrar": False,
                "producer_emitted_all_raw_linear_layer_containers": True,
                "registration_event_count": len(self.registration_receipts),
                "registration_receipts_sha256": hashlib.sha256(
                    json.dumps(
                        self.registration_receipts,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest(),
                "phase_order": list(self.phase_order),
                "fault_results": list(self.fault_results),
                "clean_captures_passed": len(self.fault_results),
                "mutants_failed_closed": len(self.fault_results),
                "formal_gpu_execution": "result-only-if-written-by-authorized-launch",
                "payload_sha256": None,
            },
            "payload_sha256",
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        observer_count = self.observer.close()
        self.registrar.close()
        if self.fault_results:
            require(observer_count == 2 * len(self.fault_results), "observer capture count drift")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.link(temporary, path)
    temporary.unlink()


def install_h20_live_binding_hooks(
    *,
    runner_module: Any,
    preregistration: Mapping[str, Any],
    capture_root: Path,
    rank: int,
    process_workers: bool = True,
) -> Callable[[], None]:
    original_witness = runner_module._run_ownership_witness_cell
    original_build = runner_module.build_resident_request_group
    original_phase = runner_module._write_witness_phase
    active: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
        "forkaudit_r40_h20_binding_context", default=None
    )
    selected = preregistration["selected_cell"]

    def witness_wrapper(*args: Any, **kwargs: Any) -> Any:
        del args
        metadata = {
            "cell_role": "ownership_witness",
            "resident_count": int(kwargs["resident_count"]),
            "kv_policy": str(kwargs["kv_policy"]),
            "gdn_base_policy": str(kwargs["gdn_base_policy"]),
            "arm_id": str(kwargs["arm_id"]),
        }
        if not all(metadata[key] == selected[key] for key in selected):
            return original_witness(**kwargs)
        state = {"metadata": metadata, "session": None}
        token = active.set(state)
        session: BindingCampaignSession | None = None
        try:
            result = original_witness(**kwargs)
            session = state["session"]
            require(isinstance(session, BindingCampaignSession), "selected witness missed session")
            payload = session.payload()
            _write_json_new(
                capture_root / f"rank-{rank}" / "raw" / "independent-live-binding.json",
                payload,
            )
            return result
        finally:
            session = state.get("session")
            if isinstance(session, BindingCampaignSession):
                session.close()
            active.reset(token)

    def build_wrapper(cache: Any, plan: Any, **kwargs: Any) -> Any:
        state = active.get()
        if state is None:
            return original_build(cache, plan, **kwargs)
        require(state["session"] is None, "selected cell built request group twice")
        session = BindingCampaignSession(
            preregistration,
            rank=rank,
            metadata=state["metadata"],
            process_workers=process_workers,
        )
        state["session"] = session
        try:
            group = original_build(cache, plan, **kwargs)
            session.register_initial(cache, group, plan)
            return group
        except BaseException:
            session.close()
            raise

    def phase_wrapper(*args: Any, **kwargs: Any) -> Any:
        state = active.get()
        if state is not None:
            session = state["session"]
            require(isinstance(session, BindingCampaignSession), "phase before registration")
            phase = str(kwargs["phase"])
            if phase in {PHASE_SETUP, PHASE_TRANSITION}:
                # This completes and acknowledges the independent challenge
                # before the immutable producer phase binder executes.
                session.run_phase(phase)
        return original_phase(*args, **kwargs)

    runner_module._run_ownership_witness_cell = witness_wrapper
    runner_module.build_resident_request_group = build_wrapper
    runner_module._write_witness_phase = phase_wrapper

    def restore() -> None:
        runner_module._run_ownership_witness_cell = original_witness
        runner_module.build_resident_request_group = original_build
        runner_module._write_witness_phase = original_phase

    return restore


__all__ = ["BindingCampaignSession", "install_h20_live_binding_hooks"]

