from __future__ import annotations

import datetime as dt
import os
import platform
import queue
from typing import Any, Callable, Mapping

import torch
import torch.multiprocessing as mp

from .candidate_binder import apply_fault_and_bind
from .detector import detect_binding
from .fixture import build_fixture
from .oracle_adapter import bind_oracle_items
from .oracle_worker import worker_main as oracle_worker_main
from .observer_worker import worker_main as observer_worker_main
from .protocol import (
    LANE_SCHEMA,
    build_manifest,
    require,
    seal_payload,
    sha256_json,
    validate_preregistration,
)


def _run_worker(
    target: Callable[[Any, Any], None], request: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    request_queue = context.Queue()
    response_queue = context.Queue()
    process = context.Process(target=target, args=(request_queue, response_queue))
    process.start()
    try:
        request_queue.put(dict(request))
        try:
            response = response_queue.get(timeout=90.0)
        except queue.Empty as exc:
            raise RuntimeError(f"{role} response timeout") from exc
        process.join(timeout=30.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10.0)
            raise RuntimeError(f"{role} process did not terminate")
        require(process.exitcode == 0, f"{role} process exit drift: {process.exitcode}")
        require(isinstance(response, dict) and response.get("ok") is True, f"{role} failed: {response}")
        return response["result"]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10.0)
        request_queue.close()
        response_queue.close()


def execute_lane(
    preregistration: Mapping[str, Any],
    fault_spec: Mapping[str, Any],
    *,
    lane_type: str,
    preregistration_sha256: str,
    source_ledger_sha256: str,
) -> dict[str, Any]:
    validate_preregistration(preregistration)
    require(lane_type in {"clean", "mutant"}, "lane type drift")
    producer_pid = os.getpid()
    manifest = build_manifest(preregistration)
    fixture = build_fixture(preregistration, fault_spec)

    oracle_items = bind_oracle_items(manifest, fixture)
    oracle = _run_worker(
        oracle_worker_main,
        {
            "manifest": manifest,
            "items": oracle_items,
            "challenge_seed_sha256": preregistration["fixture"]["challenge_seed_sha256"],
        },
        role="oracle",
    )

    candidate_items, injection = apply_fault_and_bind(
        manifest, fixture, fault_spec, lane_type=lane_type
    )
    observation = _run_worker(
        observer_worker_main,
        {
            "manifest": manifest,
            "items": candidate_items,
            "challenge_seed_sha256": preregistration["fixture"]["challenge_seed_sha256"],
        },
        role="observer",
    )
    detector = detect_binding(oracle, observation)
    required_codes = list(fault_spec["required_detection_codes"])
    expected_outcome_passed = (
        detector["passed"] is True
        if lane_type == "clean"
        else detector["passed"] is False
        and all(code in detector["failure_codes"] for code in required_codes)
    )
    expected_changed = 0 if lane_type == "clean" else int(fault_spec["expected_changed_slots"])
    acceptance_passed = all(
        (
            expected_outcome_passed,
            len(injection["changed_slot_ids"]) == expected_changed,
            injection["schema_or_label_row_mutation_used"] is False,
            injection["semantic_manifest_sha256_before"]
            == injection["semantic_manifest_sha256_after"]
            == manifest["manifest_sha256"],
            oracle["manifest_sha256"]
            == observation["manifest_sha256"]
            == manifest["manifest_sha256"],
            producer_pid != oracle["worker_pid"],
            producer_pid != observation["worker_pid"],
            oracle["worker_pid"] != observation["worker_pid"],
            oracle["parent_pid"] == producer_pid,
            observation["parent_pid"] == producer_pid,
        )
    )
    stable_oracle = {
        "manifest_sha256": oracle["manifest_sha256"],
        "rows": oracle["rows"],
        "relations": oracle["relations"],
    }
    lane = {
        "schema_version": LANE_SCHEMA,
        "experiment_id": preregistration["experiment_id"],
        "fault_id": fault_spec["fault_id"],
        "fault_kind": fault_spec["kind"],
        "lane_type": lane_type,
        "started_and_completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "producer_pid": producer_pid,
        "oracle_pid": oracle["worker_pid"],
        "observer_pid": observation["worker_pid"],
        "all_lane_roles_process_separated": len(
            {producer_pid, oracle["worker_pid"], observation["worker_pid"]}
        )
        == 3,
        "preregistration_sha256": preregistration_sha256,
        "source_ledger_sha256": source_ledger_sha256,
        "semantic_manifest": manifest,
        "stable_oracle_projection_sha256": sha256_json(stable_oracle),
        "oracle": oracle,
        "injection_receipt": injection,
        "observation": observation,
        "detector": detector,
        "required_detection_codes": required_codes,
        "expected_outcome_passed": expected_outcome_passed,
        "acceptance_passed": acceptance_passed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "fixture_device": "cpu",
        },
        "lane_payload_sha256": None,
    }
    return seal_payload(lane, "lane_payload_sha256")


__all__ = ["execute_lane"]

