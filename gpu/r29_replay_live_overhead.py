from __future__ import annotations

"""Independent offline replay for the Round-29 live-overhead evidence."""

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


RESULT_SCHEMA = "qcomem-forkaudit-r29-live-overhead-result-v1"
DESIGN_SCHEMA = "qcomem-forkaudit-r29-live-overhead-preregistration-v1"
REPLAY_SCHEMA = "qcomem-forkaudit-r29-live-overhead-replay-v1"
FULL_LAYERS = tuple(range(3, 40, 4))
LINEAR_LAYERS = tuple(index for index in range(40) if index not in FULL_LAYERS)
REQUEST_GDN_STATE_FAMILIES = ("conv", "recurrent")
REQUEST_GDN_TENSORS_PER_OWNER = len(LINEAR_LAYERS) * len(
    REQUEST_GDN_STATE_FAMILIES
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GUARD_ID_RE = re.compile(r"^[0-9a-f]{32}$")
MEASURED_SCHEDULE = (
    (0, ("baseline", "instrumented"), 0, 1),
    (1, ("instrumented", "baseline"), 1, 0),
    (2, ("baseline", "instrumented"), 1, 0),
    (3, ("instrumented", "baseline"), 0, 1),
    (4, ("baseline", "instrumented"), 0, 1),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_canonical_json(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return sha256_bytes(raw)


def load_bound_json(path: Path, expected_sha256: str, label: str) -> Any:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"{label} raw SHA drift")
    return json.loads(raw)


def safe_child(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and relative, "artifact path missing")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    require(
        candidate == resolved_root or resolved_root in candidate.parents,
        "artifact path escapes root",
    )
    return candidate


def replay_request_gdn_raw_witness(record: Any) -> dict[str, Any]:
    """Replay the serialized request-GDN binding witness without candidate code.

    The formal receipt stores the raw witness, not the derived summary returned
    by the live witness implementation's replay helper.  This implementation
    therefore validates and derives that summary directly from JSON fields.
    """

    require(type(record) is dict, "request GDN witness must be an object")
    expected_record_fields = {
        "guard_id",
        "capture_id",
        "policy",
        "layer_indices",
        "state_index",
        "resident_count",
        "completed_request_indices",
        "expected_tensor_count_per_owner",
        "rows",
        "rows_sha256",
    }
    require(
        set(record) == expected_record_fields,
        "request GDN witness schema drift",
    )
    guard_id = record["guard_id"]
    require(
        isinstance(guard_id, str) and GUARD_ID_RE.fullmatch(guard_id) is not None,
        "request GDN guard_id drift",
    )
    capture_id = record["capture_id"]
    require(
        capture_id is None
        or (
            isinstance(capture_id, str)
            and GUARD_ID_RE.fullmatch(capture_id) is not None
        ),
        "request GDN capture_id drift",
    )
    policy = record["policy"]
    require(
        policy in ("shared-base", "materialized"),
        "request GDN policy is not canonical",
    )
    layer_indices = record["layer_indices"]
    require(
        isinstance(layer_indices, list)
        and all(type(index) is int and index >= 0 for index in layer_indices)
        and len(layer_indices) == len(set(layer_indices))
        and tuple(layer_indices) == LINEAR_LAYERS,
        "request GDN layer indices drift",
    )
    state_index = record["state_index"]
    require(
        type(state_index) is int and state_index == 0,
        "request GDN state_index drift",
    )
    resident_count = record["resident_count"]
    require(
        type(resident_count) is int and resident_count >= 1,
        "request GDN resident count drift",
    )
    completed = record["completed_request_indices"]
    require(
        isinstance(completed, list)
        and all(
            type(index) is int and 0 <= index < resident_count for index in completed
        )
        and completed == sorted(set(completed)),
        "request GDN completed indices drift",
    )
    require(
        record["expected_tensor_count_per_owner"]
        == REQUEST_GDN_TENSORS_PER_OWNER,
        "request GDN tensor count drift",
    )
    rows = record["rows"]
    require(isinstance(rows, list), "request GDN rows must be a list")
    require(
        len(rows) == resident_count * REQUEST_GDN_TENSORS_PER_OWNER,
        "request GDN row cardinality drift",
    )
    rows_sha256 = record["rows_sha256"]
    require(
        isinstance(rows_sha256, str)
        and SHA256_RE.fullmatch(rows_sha256) is not None,
        "request GDN rows_sha256 drift",
    )
    require(
        sha256_canonical_json(rows) == rows_sha256,
        "request GDN row digest drift",
    )

    completed_set = set(completed)
    expected_order = [
        (request_index, layer_index, state_family, state_index)
        for request_index in range(resident_count)
        for layer_index in layer_indices
        for state_family in REQUEST_GDN_STATE_FAMILIES
    ]
    expected_row_fields = {
        "request_index",
        "layer_index",
        "state_family",
        "state_index",
        "expected_relation",
        "baseline_binding_token",
        "observed_binding_token",
        "baseline_storage_token",
        "observed_storage_token",
    }
    observed_order = []
    for row_index, row in enumerate(rows):
        require(type(row) is dict, f"request GDN row {row_index} must be an object")
        require(
            set(row) == expected_row_fields,
            f"request GDN row {row_index} schema drift",
        )
        request_index = row["request_index"]
        layer_index = row["layer_index"]
        state_family = row["state_family"]
        row_state_index = row["state_index"]
        require(
            type(request_index) is int and 0 <= request_index < resident_count,
            f"request GDN row {row_index} request index drift",
        )
        require(
            type(layer_index) is int and layer_index in layer_indices,
            f"request GDN row {row_index} layer index drift",
        )
        require(
            state_family in REQUEST_GDN_STATE_FAMILIES,
            f"request GDN row {row_index} state family drift",
        )
        require(
            type(row_state_index) is int and row_state_index == state_index,
            f"request GDN row {row_index} state index drift",
        )
        expected_relation = (
            "rebound" if request_index in completed_set else "unchanged"
        )
        require(
            row["expected_relation"] == expected_relation,
            f"request GDN row {row_index} relation drift",
        )
        token_fields = (
            "baseline_binding_token",
            "observed_binding_token",
            "baseline_storage_token",
            "observed_storage_token",
        )
        require(
            all(
                isinstance(row[field], str)
                and SHA256_RE.fullmatch(row[field]) is not None
                for field in token_fields
            ),
            f"request GDN row {row_index} token drift",
        )
        if expected_relation == "rebound":
            require(
                row["baseline_binding_token"] != row["observed_binding_token"],
                f"request GDN row {row_index} completed binding token did not change",
            )
            require(
                row["baseline_storage_token"] != row["observed_storage_token"],
                f"request GDN row {row_index} completed storage token did not change",
            )
        else:
            require(
                row["baseline_binding_token"] == row["observed_binding_token"],
                f"request GDN row {row_index} incomplete binding token changed",
            )
            require(
                row["baseline_storage_token"] == row["observed_storage_token"],
                f"request GDN row {row_index} incomplete storage token changed",
            )
        observed_order.append(
            (request_index, layer_index, state_family, row_state_index)
        )
    require(
        observed_order == expected_order,
        "request GDN row coordinate order drift",
    )
    return {
        "passed": True,
        "guard_id": guard_id,
        "capture_id": capture_id,
        "policy": policy,
        "resident_count": resident_count,
        "completed_request_indices": completed,
        "rebound_tensor_count": len(completed) * REQUEST_GDN_TENSORS_PER_OWNER,
        "unchanged_tensor_count": (resident_count - len(completed))
        * REQUEST_GDN_TENSORS_PER_OWNER,
        "rows_sha256": rows_sha256,
    }


def read_semantic_sidecar(
    artifact_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    require(
        manifest.get("schema_version")
        == "qcomem-forkaudit-r29-live-overhead-logits-v1",
        "semantic sidecar schema drift",
    )
    path = safe_child(artifact_root, manifest["path"])
    raw = path.read_bytes()
    require(len(raw) == manifest.get("bytes"), "semantic sidecar byte drift")
    require(sha256_bytes(raw) == manifest.get("sha256"), "semantic sidecar SHA drift")
    records = manifest.get("records")
    require(isinstance(records, list) and records, "semantic records missing")
    require(len(records) == manifest.get("record_count"), "semantic record count drift")
    require(manifest.get("terminal_exact_byte_coverage") is True, "coverage flag drift")
    observed: dict[str, np.ndarray] = {}
    cursor = 0
    for row in records:
        require(row.get("dtype") == "float32-le", "semantic dtype drift")
        require(row.get("offset_bytes") == cursor, "semantic offset drift")
        nbytes = row.get("nbytes")
        require(type(nbytes) is int and nbytes > 0, "semantic nbytes drift")
        segment = raw[cursor : cursor + nbytes]
        require(len(segment) == nbytes, "semantic record truncation")
        require(sha256_bytes(segment) == row.get("content_sha256"), "semantic record SHA drift")
        shape = tuple(int(item) for item in row.get("shape", ()))
        array = np.frombuffer(segment, dtype="<f4").reshape(shape).copy()
        sample_id = row.get("sample_id")
        require(isinstance(sample_id, str) and sample_id not in observed, "sample ID drift")
        require(int(array.argmax(axis=-1).item()) == row.get("token_id"), "token drift")
        observed[sample_id] = array
        cursor += nbytes
    require(cursor == len(raw), "semantic terminal coverage drift")
    return observed


def verify_capture_artifact(
    artifact_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        manifest.get("schema_version")
        == "qcomem-forkaudit-r29-live-artifact-manifest-v1",
        "audit artifact schema drift",
    )
    cell_root = safe_child(artifact_root, manifest["relative_root"])
    receipt_manifest = manifest.get("receipt", {})
    receipt_path = safe_child(cell_root, receipt_manifest.get("path"))
    receipt_raw = receipt_path.read_bytes()
    require(len(receipt_raw) == receipt_manifest.get("bytes"), "receipt byte drift")
    require(sha256_bytes(receipt_raw) == receipt_manifest.get("sha256"), "receipt SHA drift")
    receipt = json.loads(receipt_raw)
    require(
        receipt.get("schema_version")
        == "qcomem-forkaudit-r29-live-capture-receipt-v1",
        "live receipt schema drift",
    )
    binary_manifest = manifest.get("capture_binary", {})
    require(
        receipt.get("capture_binary") == binary_manifest,
        "receipt/formal binary manifest differs",
    )
    binary_path = safe_child(cell_root, binary_manifest.get("path"))
    binary_raw = binary_path.read_bytes()
    require(len(binary_raw) == binary_manifest.get("bytes"), "capture byte drift")
    require(sha256_bytes(binary_raw) == binary_manifest.get("sha256"), "capture SHA drift")
    records = binary_manifest.get("records")
    require(isinstance(records, list) and len(records) == 50, "capture record count drift")
    require(binary_manifest.get("record_count") == 50, "capture manifest count drift")
    require(binary_manifest.get("terminal_exact_byte_coverage") is True, "capture coverage flag drift")
    cursor = 0
    for record_index, row in enumerate(records):
        require(row.get("record_index") == record_index, "capture record index drift")
        require(row.get("offset_bytes") == cursor, "capture offset drift")
        nbytes = row.get("nbytes")
        require(type(nbytes) is int and nbytes > 0, "capture record size drift")
        segment = binary_raw[cursor : cursor + nbytes]
        require(len(segment) == nbytes, "capture record truncation")
        require(sha256_bytes(segment) == row.get("content_sha256"), "capture record SHA drift")
        require(row.get("layer_index") in FULL_LAYERS, "capture layer drift")
        cursor += nbytes
    require(cursor == len(binary_raw), "capture terminal coverage drift")
    append_events = receipt.get("append_events")
    call_events = receipt.get("call_events")
    require(
        isinstance(append_events, list)
        and isinstance(call_events, list)
        and len(append_events) == len(call_events) == len(FULL_LAYERS),
        "capture event cardinality drift",
    )
    require(
        tuple(row.get("layer_index") for row in append_events) == FULL_LAYERS
        and tuple(row.get("layer_index") for row in call_events) == FULL_LAYERS,
        "capture event layer order drift",
    )
    append_ids = [row.get("capture_id") for row in append_events]
    require(len(set(append_ids)) == len(append_ids), "append capture ID reused")
    require(
        [row.get("append_capture_id") for row in call_events] == append_ids,
        "call/append capture linkage drift",
    )
    live = receipt.get("live_receipts", {})
    require(live.get("source_document_immutable") is True, "source immutability missing")
    require(live.get("source_document_sha256_before") == live.get("source_document_sha256_after"), "source digest drift")
    require(live.get("ledger", {}).get("verified") is True, "ledger receipt failed")
    require(live.get("ledger", {}).get("total_calls") == 10, "ledger call count drift")
    require(live.get("kv_pre", {}).get("passed") is True, "KV pre receipt failed")
    require(live.get("kv_post", {}).get("passed") is True, "KV post receipt failed")
    request_gdn = replay_request_gdn_raw_witness(live.get("request_gdn"))
    require(
        request_gdn.get("passed") is True
        and request_gdn.get("policy") == "shared-base"
        and request_gdn.get("resident_count") == 1
        and request_gdn.get("completed_request_indices") == [0]
        and request_gdn.get("rebound_tensor_count") == 60,
        "request GDN receipt failed",
    )
    persistent_gdn = live.get("persistent_gdn", {})
    require(
        persistent_gdn.get("baseline_binding_sha256")
        == persistent_gdn.get("observed_binding_sha256")
        and persistent_gdn.get("baseline_content_sha256")
        == persistent_gdn.get("observed_content_sha256"),
        "persistent GDN receipt failed",
    )
    expected_bytes = binary_manifest["bytes"] + receipt_manifest["bytes"]
    require(manifest.get("artifact_bytes") == expected_bytes, "artifact byte sum drift")
    require(
        manifest.get("append_event_count") == 10
        and manifest.get("call_event_count") == 10
        and manifest.get("tensor_record_count") == 50,
        "formal artifact counts drift",
    )
    return {
        "sample_id": receipt.get("sample_id"),
        "capture_bytes": len(binary_raw),
        "receipt_bytes": len(receipt_raw),
        "artifact_bytes": expected_bytes,
        "tensor_record_count": len(records),
        "append_event_count": len(append_events),
        "call_event_count": len(call_events),
        "request_gdn_rebound_tensor_count": request_gdn["rebound_tensor_count"],
        "request_gdn_rows_sha256": request_gdn["rows_sha256"],
        "passed": True,
    }


def compare_semantics(
    pair: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    cells = pair.get("cells", {})
    baseline_output = cells.get("baseline", {}).get("output", {})
    instrumented_output = cells.get("instrumented", {}).get("output", {})
    baseline_id = baseline_output.get("sample_id")
    instrumented_id = instrumented_output.get("sample_id")
    require(baseline_id in arrays and instrumented_id in arrays, "semantic sample missing")
    baseline = arrays[baseline_id]
    instrumented = arrays[instrumented_id]
    require(baseline.shape == instrumented.shape, "semantic shape differs")
    exact = bool(np.array_equal(baseline, instrumented))
    baseline_token = int(baseline.argmax(axis=-1).item())
    instrumented_token = int(instrumented.argmax(axis=-1).item())
    token_equal = baseline_token == instrumented_token
    baseline_raw = baseline.astype("<f4", copy=False).tobytes(order="C")
    instrumented_raw = instrumented.astype("<f4", copy=False).tobytes(order="C")
    require(sha256_bytes(baseline_raw) == baseline_output.get("full_vocab_logit_sha256"), "baseline output SHA drift")
    require(sha256_bytes(instrumented_raw) == instrumented_output.get("full_vocab_logit_sha256"), "instrumented output SHA drift")
    recorded = pair.get("semantic_oracle", {})
    require(recorded.get("full_vocab_logits_torch_equal") is exact, "recorded exactness drift")
    require(recorded.get("generated_token_equal") is token_equal, "recorded token equality drift")
    require(recorded.get("baseline_token_id") == baseline_token, "baseline token record drift")
    require(recorded.get("instrumented_token_id") == instrumented_token, "instrumented token record drift")
    return {
        "baseline_sample_id": baseline_id,
        "instrumented_sample_id": instrumented_id,
        "full_vocab_logits_exact": exact,
        "generated_token_equal": token_equal,
        "baseline_token_id": baseline_token,
        "instrumented_token_id": instrumented_token,
    }


def recompute_summary(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(pairs) == len(MEASURED_SCHEDULE), "measured pair count drift")
    rows = []
    for expected, pair in zip(MEASURED_SCHEDULE, pairs):
        pair_index, order, baseline_slot, instrumented_slot = expected
        require(
            pair.get("pair_index") == pair_index
            and tuple(pair.get("execution_order", ())) == order
            and pair.get("baseline_slot") == baseline_slot
            and pair.get("instrumented_slot") == instrumented_slot,
            "pair schedule drift",
        )
        baseline = pair["cells"]["baseline"]
        instrumented = pair["cells"]["instrumented"]
        baseline_ns = int(baseline["wall_time_ns"])
        instrumented_ns = int(instrumented["wall_time_ns"])
        require(baseline_ns > 0 and instrumented_ns > 0, "wall time is non-positive")
        rows.append(
            {
                "pair_index": pair_index,
                "execution_order": list(order),
                "baseline_slot": baseline_slot,
                "instrumented_slot": instrumented_slot,
                "baseline_wall_time_ns": baseline_ns,
                "instrumented_wall_time_ns": instrumented_ns,
                "paired_wall_delta_ns": instrumented_ns - baseline_ns,
                "paired_wall_ratio": instrumented_ns / baseline_ns,
                "baseline_incremental_peak_allocated_bytes": int(
                    baseline["incremental_peak_allocated_bytes"]
                ),
                "instrumented_incremental_peak_allocated_bytes": int(
                    instrumented["incremental_peak_allocated_bytes"]
                ),
                "paired_incremental_peak_delta_bytes": int(
                    instrumented["incremental_peak_allocated_bytes"]
                )
                - int(baseline["incremental_peak_allocated_bytes"]),
                "instrumented_audit_artifact_bytes": int(
                    instrumented["audit_artifact_bytes"]
                ),
                "pair_valid": bool(pair["pair_valid"]),
            }
        )
    return {
        "measured_pair_count": len(rows),
        "warmup_pairs_included": 0,
        "rows": rows,
        "median_paired_wall_delta_ns": statistics.median(
            row["paired_wall_delta_ns"] for row in rows
        ),
        "min_paired_wall_delta_ns": min(row["paired_wall_delta_ns"] for row in rows),
        "max_paired_wall_delta_ns": max(row["paired_wall_delta_ns"] for row in rows),
        "median_paired_wall_ratio": statistics.median(
            row["paired_wall_ratio"] for row in rows
        ),
        "median_paired_incremental_peak_delta_bytes": statistics.median(
            row["paired_incremental_peak_delta_bytes"] for row in rows
        ),
        "median_instrumented_audit_artifact_bytes": statistics.median(
            row["instrumented_audit_artifact_bytes"] for row in rows
        ),
        "negative_numeric_deltas_preserved": True,
        "statistical_significance_claimed": False,
    }


def compare_summaries(recorded: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    exact_fields = (
        "measured_pair_count",
        "warmup_pairs_included",
        "rows",
        "median_paired_wall_delta_ns",
        "min_paired_wall_delta_ns",
        "max_paired_wall_delta_ns",
        "median_paired_incremental_peak_delta_bytes",
        "median_instrumented_audit_artifact_bytes",
        "negative_numeric_deltas_preserved",
        "statistical_significance_claimed",
    )
    for field in exact_fields:
        require(recorded.get(field) == observed.get(field), f"summary {field} drift")
    require(
        math.isclose(
            float(recorded.get("median_paired_wall_ratio")),
            float(observed.get("median_paired_wall_ratio")),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "summary median ratio drift",
    )


def replay_result(
    design: Mapping[str, Any],
    result: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    require(design.get("schema_version") == DESIGN_SCHEMA, "design schema drift")
    require(result.get("schema_version") == RESULT_SCHEMA, "result schema drift")
    require(result.get("status") == "completed", "formal result incomplete")
    arrays = read_semantic_sidecar(artifact_root, result.get("semantic_sidecar", {}))
    warmup = result.get("warmup_pair")
    measured = result.get("measured_pairs")
    require(isinstance(warmup, dict) and isinstance(measured, list), "pair payload missing")
    require(warmup.get("warmup") is True, "warmup label drift")
    require(tuple(warmup.get("execution_order", ())) == ("instrumented", "baseline"), "warmup order drift")
    all_pairs = [warmup, *measured]
    semantic_rows = []
    artifact_rows = []
    recomputed_pair_valid = []
    for pair in all_pairs:
        baseline = pair.get("cells", {}).get("baseline", {})
        instrumented = pair.get("cells", {}).get("instrumented", {})
        require(
            baseline.get("capture_policy") == "optional-forkaudit-capture-disabled"
            and baseline.get("append_observer_enabled") is False
            and baseline.get("call_observer_enabled") is False
            and baseline.get("ownership_receipt_enabled") is False
            and baseline.get("audit_artifact_manifest") is None
            and baseline.get("audit_artifact_bytes") == 0,
            "baseline capture-disabled contract drift",
        )
        require(
            instrumented.get("capture_policy")
            == "full-live-capture-and-ownership-receipt"
            and instrumented.get("append_observer_enabled") is True
            and instrumented.get("call_observer_enabled") is True
            and instrumented.get("ownership_receipt_enabled") is True,
            "instrumented capture contract drift",
        )
        baseline_ledger = baseline.get("ledger", {})
        instrumented_ledger = instrumented.get("ledger", {})
        require(
            baseline_ledger.get("implementation") == "MultiForkHitLedger"
            and instrumented_ledger.get("implementation") == "MultiForkHitLedger"
            and baseline_ledger.get("explicit_frozen_kernel") is True
            and instrumented_ledger.get("explicit_frozen_kernel") is True
            and baseline_ledger.get("call_observer_enabled") is False
            and instrumented_ledger.get("call_observer_enabled") is True
            and baseline_ledger.get("kernel_identity")
            == instrumented_ledger.get("kernel_identity")
            and bool(baseline_ledger.get("kernel_identity")),
            "baseline/instrumented common-ledger or frozen-kernel parity drift",
        )
        for arm in (baseline, instrumented):
            require(
                type(arm.get("wall_time_ns")) is int
                and arm["wall_time_ns"] > 0
                and type(arm.get("peak_allocated_bytes")) is int
                and arm["peak_allocated_bytes"] >= arm["allocated_before_bytes"]
                and arm["incremental_peak_allocated_bytes"]
                == arm["peak_allocated_bytes"] - arm["allocated_before_bytes"]
                and arm.get("cuda_synchronized_before_start") is True
                and arm.get("cuda_synchronized_before_stop") is True
                and arm.get("peak_stats_reset_before_start") is True,
                "timing/allocator contract drift",
            )
        artifact = verify_capture_artifact(
            artifact_root,
            instrumented.get("audit_artifact_manifest", {}),
        )
        require(
            instrumented.get("audit_artifact_bytes") == artifact["artifact_bytes"],
            "instrumented artifact bytes drift",
        )
        semantic = compare_semantics(pair, arrays)
        pair_valid = (
            pair.get("source_document_immutable") is True
            and pair.get("persistent_gdn_immutable") is True
            and semantic["full_vocab_logits_exact"]
            and semantic["generated_token_equal"]
            and artifact["passed"]
        )
        require(pair.get("pair_valid") is pair_valid, "recorded pair validity drift")
        semantic_rows.append({"pair_label": pair.get("pair_label"), **semantic})
        artifact_rows.append({"pair_label": pair.get("pair_label"), **artifact})
        recomputed_pair_valid.append(pair_valid)
    require(len(arrays) == 2 * len(all_pairs), "semantic sample cardinality drift")
    observed_summary = recompute_summary(measured)
    compare_summaries(result.get("paired_summary", {}), observed_summary)
    scientific_valid = all(recomputed_pair_valid)
    require(
        result.get("scientific_run_valid") is scientific_valid
        and result.get("formal_evidence_eligible") is scientific_valid,
        "formal validity label drift",
    )
    validity = result.get("validity", {})
    require(
        validity.get("warmup_pair_count") == 1
        and validity.get("warmup_discarded_from_estimands") is True
        and validity.get("measured_pair_count") == 5
        and validity.get("alternating_schedule_verified") is True
        and validity.get("negative_numeric_deltas_removed") is False,
        "formal validity receipt drift",
    )
    return {
        "schema_version": REPLAY_SCHEMA,
        "replay_passed": True,
        "scientific_run_valid_recomputed": scientific_valid,
        "formal_evidence_eligible_recomputed": scientific_valid,
        "semantic_rows": semantic_rows,
        "artifact_rows": artifact_rows,
        "paired_summary_recomputed": observed_summary,
        "warmup_excluded_from_estimands": True,
        "negative_numeric_deltas_preserved": True,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--design-preregistration", type=Path, required=True)
    result.add_argument("--expected-design-sha256", required=True)
    result.add_argument("--formal-result", type=Path, required=True)
    result.add_argument("--expected-formal-result-sha256", required=True)
    result.add_argument("--artifact-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    design = load_bound_json(
        args.design_preregistration,
        args.expected_design_sha256,
        "design preregistration",
    )
    result = load_bound_json(
        args.formal_result,
        args.expected_formal_result_sha256,
        "formal result",
    )
    value = replay_result(design, result, args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), "replay output path already exists")
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
