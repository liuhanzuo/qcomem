#!/usr/bin/env python3
"""Bind Round-39 compiled receipts to one valid Round-29 formal execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from r39_compiled_dispatch_receipts import (
    DispatchReceiptError,
    SCHEMA_VERSION,
    TARGET_VLLM_ENTRYPOINT,
    _sha256_file,
    _write_json,
    verify_payload,
)


R29_SCHEMA = "qcomem-forkaudit-r29-live-overhead-result-v1"
R29_PROTOCOL = "qcomem-forkaudit-paired-live-request-overhead-v1"
R29_DESIGN_SHA256 = "2114d2cd85bedc1eafa5d1398fd0afd0d57819c0360c3be3f9ec20f1b2878939"
R39_AGGREGATE_SCHEMA = "forkaudit-r39-compiled-dispatch-formal-binding-v2"
EXPECTED_CONTROL_NAMES = {
    "receipt-config-tamper",
    "receipt-artifact-id-tamper",
    "missing-required-ptx",
    "extra-unreceipted-artifact",
    "compiled-artifact-substitution",
    "gdn-runtime-source-substitution",
    "gdn-cache-rebind-source-substitution",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DispatchReceiptError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DispatchReceiptError(f"cannot read {label}: {error}") from error


def _verify_semantic_sidecar(result: Mapping[str, Any], path: Path) -> dict[str, Any]:
    sidecar = _mapping(result.get("semantic_sidecar"), "R29 semantic sidecar")
    _require(sidecar.get("terminal_exact_byte_coverage") is True, "R29 sidecar lacks terminal coverage")
    _require(sidecar.get("sha256") == _sha256_file(path), "R29 semantic sidecar hash mismatch")
    raw = path.read_bytes()
    _require(sidecar.get("bytes") == len(raw), "R29 semantic sidecar byte count mismatch")
    records = sidecar.get("records")
    _require(isinstance(records, list) and records, "R29 semantic sidecar records missing")
    _require(sidecar.get("record_count") == len(records), "R29 semantic record count mismatch")
    expected_offset = 0
    sample_ids: list[str] = []
    for index, row_raw in enumerate(records):
        row = _mapping(row_raw, f"R29 semantic record {index}")
        offset = row.get("offset_bytes")
        nbytes = row.get("nbytes")
        _require(type(offset) is int and offset == expected_offset, "R29 semantic sidecar has a gap/overlap")
        _require(type(nbytes) is int and nbytes > 0, "R29 semantic record byte count invalid")
        payload = raw[offset : offset + nbytes]
        _require(len(payload) == nbytes, "R29 semantic record exceeds sidecar")
        _require(hashlib.sha256(payload).hexdigest() == row.get("content_sha256"), "R29 semantic record hash mismatch")
        sample_id = row.get("sample_id")
        _require(isinstance(sample_id, str) and sample_id, "R29 semantic sample id missing")
        sample_ids.append(sample_id)
        expected_offset += nbytes
    _require(expected_offset == len(raw), "R29 semantic records do not cover the sidecar exactly")
    _require(len(set(sample_ids)) == len(sample_ids), "R29 semantic sample ids are not unique")
    return {
        "sha256": _sha256_file(path),
        "bytes": len(raw),
        "record_count": len(records),
        "terminal_exact_byte_coverage": True,
    }


def _verify_r29_result(
    result: Mapping[str, Any], *, semantic_sidecar: Path, design: Path
) -> dict[str, Any]:
    _require(result.get("schema_version") == R29_SCHEMA, "R29 result schema drift")
    _require(result.get("protocol") == R29_PROTOCOL, "R29 protocol drift")
    _require(result.get("status") == "completed", "R29 result is not complete")
    for field in (
        "formal_evidence_eligible",
        "scientific_execution_completed",
        "scientific_run_valid",
    ):
        _require(result.get(field) is True, f"R29 {field} is not true")
    _require(_sha256_file(design) == R29_DESIGN_SHA256, "R29 design file hash drift")
    _require(
        result.get("design_preregistration_raw_sha256") == R29_DESIGN_SHA256,
        "R29 result is not bound to the frozen design",
    )
    validity = _mapping(result.get("validity"), "R29 validity")
    for field in (
        "warmup_discarded_from_estimands",
        "alternating_schedule_verified",
        "all_pair_semantic_oracles_exact",
        "all_live_receipts_valid",
    ):
        _require(validity.get(field) is True, f"R29 validity.{field} is not true")
    _require(validity.get("warmup_pair_count") == 1, "R29 warmup count drift")
    _require(validity.get("measured_pair_count") == 5, "R29 measured-pair count drift")

    warmup = _mapping(result.get("warmup_pair"), "R29 warmup pair")
    measured = result.get("measured_pairs")
    _require(isinstance(measured, list) and len(measured) == 5, "R29 measured pairs missing")
    pairs: list[Mapping[str, Any]] = [warmup, *[_mapping(item, "R29 measured pair") for item in measured]]
    _require(warmup.get("pair_label") == "warmup-pair", "R29 warmup label drift")
    _require(warmup.get("warmup") is True and warmup.get("discarded_from_estimands") is True, "R29 warmup policy drift")

    environment = _mapping(result.get("environment"), "R29 environment")
    geometry = _mapping(environment.get("model_geometry"), "R29 model geometry")
    observed = _mapping(geometry.get("observed"), "R29 observed model geometry")
    _require(geometry.get("matches_frozen_geometry") is True, "R29 model geometry is not frozen")
    _require(geometry.get("num_hidden_layers") == 40, "R29 hidden-layer count drift")
    full_layers = observed.get("full_attention_layers")
    linear_layers = geometry.get("linear_attention_layer_count")
    _require(full_layers == 10 and linear_layers == 30, "R29 hybrid layer geometry drift")
    _require(full_layers + linear_layers == geometry.get("num_hidden_layers"), "R29 layer partition is incomplete")
    full_layer_indices = geometry.get("full_attention_layer_indices")
    _require(
        full_layer_indices == list(range(3, 40, 4)),
        "R29 full-attention layer indices drift",
    )
    linear_layer_indices = [
        index for index in range(40) if index not in set(full_layer_indices)
    ]
    _require(len(linear_layer_indices) == linear_layers, "R29 linear-layer set drift")
    kernel_identity = _mapping(environment.get("kernel_identity"), "R29 kernel identity")
    _require(
        f"{kernel_identity.get('module')}.{kernel_identity.get('qualname')}" == TARGET_VLLM_ENTRYPOINT,
        "R29 kernel identity differs from the intercepted entrypoint",
    )

    expected_attention_calls = 0
    expected_document_prefill_gdn_calls = 0
    expected_request_cell_gdn_calls = 0
    call_ranges: list[dict[str, Any]] = []
    next_attention = 0
    next_gdn = 0
    expected_measured_orders = [
        ["baseline", "instrumented"],
        ["instrumented", "baseline"],
        ["baseline", "instrumented"],
        ["instrumented", "baseline"],
        ["baseline", "instrumented"],
    ]
    for pair_position, pair in enumerate(pairs):
        if pair_position:
            measured_index = pair_position - 1
            _require(pair.get("pair_label") == f"measured-pair-{measured_index}", "R29 measured-pair label drift")
            _require(pair.get("pair_index") == measured_index, "R29 measured-pair index drift")
            _require(pair.get("warmup") is False and pair.get("discarded_from_estimands") is False, "R29 measured-pair policy drift")
            _require(pair.get("execution_order") == expected_measured_orders[measured_index], "R29 alternating order drift")
        else:
            _require(pair.get("execution_order") == ["instrumented", "baseline"], "R29 warmup order drift")
        _require(pair.get("pair_valid") is True, "R29 pair is invalid")
        _require(pair.get("source_document_immutable") is True, "R29 source document changed")
        _require(pair.get("persistent_gdn_immutable") is True, "R29 persistent GDN state changed")
        conversion = _mapping(pair.get("conversion"), "R29 pair conversion")
        _require(
            conversion.get("document_length") == 4033
            and conversion.get("max_append_tokens") == 16
            and conversion.get("full_attention_layer_count") == full_layers,
            "R29 pair document/query geometry drift",
        )
        prefill_start = next_gdn
        next_gdn += linear_layers
        expected_document_prefill_gdn_calls += linear_layers
        prefill_range = [prefill_start, next_gdn]
        semantic = _mapping(pair.get("semantic_oracle"), "R29 semantic oracle")
        _require(semantic.get("full_vocab_logits_torch_equal") is True, "R29 full logits differ")
        _require(semantic.get("generated_token_equal") is True, "R29 generated token differs")
        _require(semantic.get("max_abs_error") == 0.0 and semantic.get("mean_abs_error") == 0.0, "R29 semantic error is nonzero")
        _require(semantic.get("baseline_sha256") == semantic.get("instrumented_sha256"), "R29 semantic hashes differ")
        cells = _mapping(pair.get("cells"), "R29 cells")
        _require(set(cells) == {"baseline", "instrumented"}, "R29 pair cell set drift")
        execution_order = pair.get("execution_order")
        _require(isinstance(execution_order, list) and set(execution_order) == set(cells), "R29 cell execution order invalid")
        for arm in execution_order:
            cell = _mapping(cells[arm], f"R29 {arm} cell")
            ledger = _mapping(cell.get("ledger"), f"R29 {arm} ledger")
            _require(ledger.get("verified") is True, "R29 attention ledger is unverified")
            _require(ledger.get("explicit_frozen_kernel") is True, "R29 kernel was not frozen")
            _require(ledger.get("dense_fallback_calls") == 0, "R29 dense fallback was used")
            _require(ledger.get("full_kv_concatenations") == 0, "R29 full-KV concatenation was used")
            _require(ledger.get("total_calls") == full_layers, "R29 attention call count differs from geometry")
            _require(ledger.get("call_observer_enabled") is (arm == "instrumented"), "R29 observer arm binding drift")
            ledger_kernel = _mapping(ledger.get("kernel_identity"), "R29 ledger kernel identity")
            _require(dict(ledger_kernel) == dict(kernel_identity), "R29 cell kernel identity drift")
            attention_start = next_attention
            gdn_start = next_gdn
            next_attention += full_layers
            next_gdn += linear_layers
            expected_attention_calls += full_layers
            expected_request_cell_gdn_calls += linear_layers
            call_ranges.append(
                {
                    "pair_label": pair.get("pair_label"),
                    "arm": arm,
                    "document_prefill_gdn_call_range": prefill_range,
                    "attention_call_range": [attention_start, next_attention],
                    "gdn_call_range": [gdn_start, next_gdn],
                }
            )

    sidecar_summary = _verify_semantic_sidecar(result, semantic_sidecar)
    _require(sidecar_summary["record_count"] == len(call_ranges), "R29 semantic sidecar/cell count mismatch")
    expected_gdn_calls = (
        expected_document_prefill_gdn_calls + expected_request_cell_gdn_calls
    )
    _require(
        expected_document_prefill_gdn_calls == 6 * 30
        and expected_request_cell_gdn_calls == 12 * 30
        and expected_gdn_calls == 540,
        "R29 GDN phase-count derivation drift",
    )
    return {
        "schema_version": R29_SCHEMA,
        "design_sha256": R29_DESIGN_SHA256,
        "pair_count": len(pairs),
        "cell_count": len(call_ranges),
        "expected_attention_calls": expected_attention_calls,
        "expected_gdn_calls": expected_gdn_calls,
        "expected_document_prefill_gdn_calls": expected_document_prefill_gdn_calls,
        "expected_request_cell_gdn_calls": expected_request_cell_gdn_calls,
        "linear_layer_indices": linear_layer_indices,
        "call_ranges_half_open": call_ranges,
        "semantic_sidecar": sidecar_summary,
    }


def verify_formal_binding(
    *,
    r29_result_path: Path,
    semantic_sidecar: Path,
    design: Path,
    receipt_path: Path,
    replay_path: Path,
    controls_path: Path,
    cache_root: Path,
    code_root: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    r29_raw = _mapping(_load_json(r29_result_path, "R29 result"), "R29 result")
    r29_summary = _verify_r29_result(r29_raw, semantic_sidecar=semantic_sidecar, design=design)
    receipt = _load_json(receipt_path, "compiled-dispatch receipt")
    replay_observed = _mapping(_load_json(replay_path, "detached replay"), "detached replay")
    replay_expected = verify_payload(
        receipt,
        cache_root=cache_root,
        code_root=code_root,
        runtime_root=runtime_root,
    )
    _require(dict(replay_observed) == replay_expected, "stored detached replay differs from fresh replay")
    _require(replay_expected.get("replay_verdict") == "pass", "compiled-dispatch replay failed")
    _require(
        replay_expected.get("attention_call_count") == r29_summary["expected_attention_calls"],
        "compiled attention receipt count does not close over R29 cells",
    )
    _require(
        replay_expected.get("gdn_call_count") == r29_summary["expected_gdn_calls"],
        "GDN source receipt count does not close over R29 cells",
    )
    _require(
        replay_expected.get("gdn_document_prefill_call_count")
        == r29_summary["expected_document_prefill_gdn_calls"],
        "GDN document-prefill receipt count does not close over six R29 pairs",
    )
    _require(
        replay_expected.get("gdn_request_cell_call_count")
        == r29_summary["expected_request_cell_gdn_calls"],
        "GDN request-cell receipt count does not close over twelve R29 cells",
    )
    receipt_mapping = _mapping(receipt, "compiled-dispatch receipt")
    for index, row_raw in enumerate(receipt_mapping.get("attention_calls", [])):
        row = _mapping(row_raw, f"attention receipt {index}")
        shape = _mapping(row.get("call_shape"), f"attention call shape {index}")
        _require({"q", "k", "v", "out", "block_table"}.issubset(shape), "attention call shape is incomplete")

    gdn_rows_raw = receipt_mapping.get("gdn_calls")
    _require(isinstance(gdn_rows_raw, list), "GDN receipt list missing")
    gdn_rows = [_mapping(row, "GDN receipt") for row in gdn_rows_raw]
    linear_indices = r29_summary["linear_layer_indices"]
    ranges = r29_summary["call_ranges_half_open"]
    _require(len(ranges) == 12, "R29 call-range count drift")
    cursor = 0
    for pair_position in range(6):
        pair_ranges = ranges[pair_position * 2 : pair_position * 2 + 2]
        prefill_range = pair_ranges[0]["document_prefill_gdn_call_range"]
        _require(
            all(
                row["document_prefill_gdn_call_range"] == prefill_range
                for row in pair_ranges
            ),
            "R29 pair arms disagree on document-prefill range",
        )
        _require(prefill_range == [cursor, cursor + 30], "GDN prefill range drift")
        prefill_rows = gdn_rows[cursor : cursor + 30]
        _require(len(prefill_rows) == 30, "GDN prefill rows missing")
        _require(
            [row.get("layer_idx") for row in prefill_rows] == linear_indices,
            "GDN prefill layer order/coverage drift",
        )
        _require(
            all(
                row.get("execution_phase") == "document-prefill"
                and row.get("cache_has_previous_state") is False
                and row.get("sequence_length") == 4033
                for row in prefill_rows
            ),
            "GDN document-prefill phase binding drift",
        )
        cursor += 30
        for cell_range in pair_ranges:
            _require(
                cell_range["gdn_call_range"] == [cursor, cursor + 30],
                "GDN request-cell range drift",
            )
            request_rows = gdn_rows[cursor : cursor + 30]
            _require(len(request_rows) == 30, "GDN request-cell rows missing")
            _require(
                [row.get("layer_idx") for row in request_rows] == linear_indices,
                "GDN request-cell layer order/coverage drift",
            )
            _require(
                all(
                    row.get("execution_phase") == "request-cell"
                    and row.get("cache_has_previous_state") is True
                    and row.get("sequence_length") == 16
                    for row in request_rows
                ),
                "GDN request-cell phase binding drift",
            )
            cursor += 30
    _require(cursor == len(gdn_rows) == 540, "GDN phase sequence has extra/missing rows")

    controls = _mapping(_load_json(controls_path, "bound negative controls"), "bound negative controls")
    _require(controls.get("control_basis") == "actual-captured-receipt-and-artifacts", "controls did not use actual artifacts")
    _require(controls.get("receipt_sha256") == _sha256_file(receipt_path), "controls bind a different receipt")
    _require(controls.get("all_rejected") is True, "one or more bound controls passed")
    control_rows = _mapping(controls.get("negative_controls"), "negative-control rows")
    _require(set(control_rows) == EXPECTED_CONTROL_NAMES, "bound negative-control set drift")
    _require(all(value == "rejected" for value in control_rows.values()), "bound negative control not rejected")

    artifact_ids = sorted(
        {
            _mapping(row, "attention receipt")["selected_compiled_artifact"]["artifact_id"]
            for row in receipt_mapping["attention_calls"]
        }
    )
    configurations = sorted(
        {
            json.dumps(
                _mapping(row, "attention receipt")["selected_compile_config"],
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in receipt_mapping["attention_calls"]
        }
    )
    return {
        "schema_version": R39_AGGREGATE_SCHEMA,
        "status": "pass",
        "formal_evidence_eligible": True,
        "r29_execution_binding": r29_summary,
        "compiled_dispatch_replay": replay_expected,
        "compiled_artifact_ids": artifact_ids,
        "selected_compile_configurations": [json.loads(item) for item in configurations],
        "bound_negative_controls": dict(controls),
        "input_sha256": {
            "r29_formal_result": _sha256_file(r29_result_path),
            "r29_semantic_sidecar": _sha256_file(semantic_sidecar),
            "r29_design_preregistration": _sha256_file(design),
            "compiled_dispatch_receipt": _sha256_file(receipt_path),
            "detached_replay": _sha256_file(replay_path),
            "bound_negative_controls": _sha256_file(controls_path),
        },
        "claim_boundary": {
            "established": "For this exact valid R29 H20 execution, every expected vLLM unified-attention call is bound to its fully hashed selected Triton artifact/configuration; all 180 document-prefill and 360 request-cell native Transformers GDN calls are separately closed over the selected eager torch_chunk_gated_delta_rule and qcomem functional cache-rebind source hashes.",
            "not_established": [
                "compiled dispatch for the eager Transformers GDN path or its underlying ATen/CUDA operators",
                "runtime attestation or malicious-producer resistance",
                "cross-model, cross-runtime, or cross-hardware generality",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r29-result", type=Path, required=True)
    parser.add_argument("--semantic-sidecar", type=Path, required=True)
    parser.add_argument("--design-preregistration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--negative-controls", type=Path, required=True)
    parser.add_argument("--triton-cache-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = verify_formal_binding(
        r29_result_path=args.r29_result,
        semantic_sidecar=args.semantic_sidecar,
        design=args.design_preregistration,
        receipt_path=args.receipt,
        replay_path=args.replay,
        controls_path=args.negative_controls,
        cache_root=args.triton_cache_root,
        code_root=args.code_root,
        runtime_root=args.runtime_root,
    )
    _write_json(args.output, aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
