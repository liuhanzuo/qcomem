#!/usr/bin/env python3
"""Build the R40 conventional-baseline versus ForkAudit matrix.

This program is deliberately read-only outside its requested output directory.
It consumes archived JSON evidence, recomputes the frozen plain checks, validates
the bound RR2 shards, and emits deterministic machine-readable products.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ANALYSIS_ID = "R40-BASELINE-DETECTOR-MATRIX-V1"
DETECTOR_IDS = (
    "B1_RUNTIME_ASSERTION",
    "B2_TOKEN_EQUALITY",
    "B3_FULL_LOGIT_EQUALITY",
    "B4_STRUCTURAL_SEQUENCE_ARGUMENT",
    "B5_PERSISTENT_BASE_IMMUTABILITY",
    "B6_SIMPLE_ALIAS_OVERLAP",
    "B7_BASIC_LIFECYCLE_CARDINALITY",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


class EvidenceTracker:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.entries: dict[str, dict[str, Any]] = {}

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.repo_root).as_posix()

    def register(self, path: Path, role: str) -> None:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = self.relative(path)
        entry = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "roles": [],
        }
        if relative in self.entries:
            entry = self.entries[relative]
        if role not in entry["roles"]:
            entry["roles"].append(role)
            entry["roles"].sort()
        self.entries[relative] = entry

    def load_json(self, path: Path, role: str) -> Any:
        self.register(path, role)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def manifest(self) -> dict[str, Any]:
        entries = [self.entries[key] for key in sorted(self.entries)]
        return {
            "schema_version": "forkaudit-r40-baseline-input-manifest-v1",
            "analysis_id": ANALYSIS_ID,
            "file_count": len(entries),
            "files": entries,
        }


def decision(
    detector_id: str,
    *,
    status: str,
    caught: bool | None,
    evidence_mode: str,
    location: str | None,
    source_paths: Iterable[str],
    detail: Any,
) -> dict[str, Any]:
    if detector_id not in DETECTOR_IDS:
        raise ValueError(f"unknown detector: {detector_id}")
    if status == "evaluated" and type(caught) is not bool:
        raise ValueError(f"evaluated detector {detector_id} lacks boolean caught")
    if status != "evaluated" and caught is not None:
        raise ValueError(f"non-evaluated detector {detector_id} has caught={caught!r}")
    return {
        "detector_id": detector_id,
        "status": status,
        "caught": caught,
        "evidence_mode": evidence_mode,
        "first_localization_if_caught": location if caught else None,
        "source_paths": sorted(set(source_paths)),
        "detail": detail,
    }


def unavailable(detector_id: str, status: str, source_paths: Iterable[str], detail: str) -> dict[str, Any]:
    return decision(
        detector_id,
        status=status,
        caught=None,
        evidence_mode=status,
        location=None,
        source_paths=source_paths,
        detail=detail,
    )


def detector_map(decisions: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped = {item["detector_id"]: item for item in decisions}
    if set(mapped) != set(DETECTOR_IDS):
        raise ValueError(f"detector coverage drift: {sorted(mapped)}")
    return mapped


def earliest_catch(
    decisions: dict[str, dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any] | None:
    stage_order = protocol["stage_order"]
    tie_break = {detector_id: index for index, detector_id in enumerate(protocol["detector_tie_break"])}
    candidates = []
    for detector_id, item in decisions.items():
        if item["caught"] is not True:
            continue
        stage = item["first_localization_if_caught"]["stage"]
        candidates.append((stage_order[stage], tie_break[detector_id], detector_id, item))
    if not candidates:
        return None
    _, _, detector_id, item = min(candidates)
    return {
        "detector_id": detector_id,
        "location": item["first_localization_if_caught"],
        "evidence_mode": item["evidence_mode"],
    }


def relation_label(case_kind: str, baseline_detected: bool, forkaudit_detected: bool) -> str:
    if case_kind == "clean":
        return "clean_false_positive" if (baseline_detected or forkaudit_detected) else "clean_pass"
    if baseline_detected and forkaudit_detected:
        return "redundant_both"
    if forkaudit_detected:
        return "forkaudit_unique"
    if baseline_detected:
        return "baseline_only"
    return "neither"


def finalize_row(
    *,
    campaign: str,
    case_id: str,
    case_kind: str,
    fault_id: str | None,
    coordinate: str | None,
    baseline_decisions: Iterable[dict[str, Any]],
    forkaudit_detected: bool,
    forkaudit_first_predicate: str | None,
    forkaudit_location: dict[str, Any] | None,
    forkaudit_source_paths: Iterable[str],
    execution_relationship: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    mapped = detector_map(baseline_decisions)
    caught_ids = sorted(detector_id for detector_id, item in mapped.items() if item["caught"] is True)
    strict_caught_ids = sorted(
        detector_id
        for detector_id, item in mapped.items()
        if item["caught"] is True and item["evidence_mode"] == "executed_independent_observer"
    )
    baseline_detected = bool(caught_ids)
    return {
        "campaign": campaign,
        "case_id": case_id,
        "case_kind": case_kind,
        "fault_id": fault_id,
        "coordinate": coordinate,
        "execution_relationship": execution_relationship,
        "baseline": {
            "detected": baseline_detected,
            "strict_independent_baseline_detected": bool(strict_caught_ids),
            "caught_detector_ids": caught_ids,
            "strict_independent_caught_detector_ids": strict_caught_ids,
            "first_localization": earliest_catch(mapped, protocol),
            "detectors": mapped,
        },
        "forkaudit": {
            "detected": forkaudit_detected,
            "first_failed_predicate": forkaudit_first_predicate,
            "first_localization": forkaudit_location,
            "source_paths": sorted(set(forkaudit_source_paths)),
        },
        "catch_relation": relation_label(case_kind, baseline_detected, forkaudit_detected),
    }


def primary_forkaudit_location(predicate: str | None) -> dict[str, Any] | None:
    if predicate is None:
        return None
    stages = {
        "KV_RESERVATION_DISJOINT": ("pre_model_validation", "request reservation/constructor binding"),
        "KV_SEQUENCE_ID": ("dispatch_or_transition", "request sequence binding"),
        "KV_TAIL_COW": ("dispatch_or_transition", "tail detach before append"),
        "gdn_completed_vs_base_disjoint": ("post_transition_state", "completed request versus persistent base"),
        "gdn_completed_vs_peers_disjoint": ("post_transition_state", "completed request versus peer"),
        "POSITION_CANONICAL_VALUES": ("dispatch_or_transition", "post-RoPE position values"),
        "MASK_CONTRACT": ("dispatch_or_transition", "attention mask contract"),
        "KERNEL_CALLABLE_ID": ("dispatch_or_transition", "attention callable identity"),
        "KV_PAGED_VIEW": ("dispatch_or_transition", "paged-view representation"),
    }
    stage, scope = stages[predicate]
    return {"stage": stage, "scope": scope}


def primary_baseline_decisions(
    case: dict[str, Any],
    raw_path: str,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    outcome = case["outcome"]
    production = outcome["production"]
    semantics = outcome["semantics"]
    decisions: list[dict[str, Any]] = []

    assertion = production["assertion"]
    crash = production["nonassertion_crash"]
    evaluated_runtime = assertion["status"] == "evaluated" or crash["status"] == "evaluated"
    if evaluated_runtime:
        caught = assertion.get("caught") is True or crash.get("caught") is True
        decisions.append(
            decision(
                "B1_RUNTIME_ASSERTION",
                status="evaluated",
                caught=caught,
                evidence_mode="executed_independent_observer",
                location={"stage": "dispatch_or_transition", "scope": "production assertion/runtime"},
                source_paths=[raw_path],
                detail={
                    "production_assertion": assertion,
                    "nonassertion_crash": crash,
                    "fault_payload_abort_excluded": production["fault_payload_abort"],
                },
            )
        )
    else:
        decisions.append(unavailable("B1_RUNTIME_ASSERTION", "not_evaluated", [raw_path], "execution ended before production observer"))

    for detector_id, key, stage in (
        ("B2_TOKEN_EQUALITY", "token_only", "terminal_token"),
        ("B3_FULL_LOGIT_EQUALITY", "full_logit", "terminal_logit"),
    ):
        item = semantics[key]
        if item["status"] == "evaluated":
            decisions.append(
                decision(
                    detector_id,
                    status="evaluated",
                    caught=item["caught"],
                    evidence_mode="executed_independent_observer",
                    location={"stage": stage, "scope": key.replace("_", " ")},
                    source_paths=[raw_path],
                    detail=item,
                )
            )
        else:
            decisions.append(unavailable(detector_id, "not_evaluated", [raw_path], "matched output observer unavailable"))

    events = outcome["fork_audit"]["target_suppression_events"]
    gate_map = protocol["primary_projected_gate_map"]
    expected_gate = case["expected_gate_id"]
    projected_detector = gate_map.get(expected_gate)
    projected_details: dict[str, list[Any]] = defaultdict(list)
    for event in events:
        detector_id = gate_map.get(event["gate_id"])
        if detector_id is not None:
            projected_details[detector_id].append(event)

    for detector_id in (
        "B4_STRUCTURAL_SEQUENCE_ARGUMENT",
        "B6_SIMPLE_ALIAS_OVERLAP",
        "B7_BASIC_LIFECYCLE_CARDINALITY",
    ):
        relevant = projected_detector == detector_id or bool(projected_details[detector_id])
        if relevant:
            location_scope = {
                "B4_STRUCTURAL_SEQUENCE_ARGUMENT": "plain structural/sequence assertion",
                "B6_SIMPLE_ALIAS_OVERLAP": "plain write-ready alias/overlap assertion",
                "B7_BASIC_LIFECYCLE_CARDINALITY": "plain tail/lifecycle assertion",
            }[detector_id]
            stage = "pre_model_validation" if expected_gate == "KV_RESERVATION_DISJOINT" else (
                "post_transition_state" if detector_id == "B6_SIMPLE_ALIAS_OVERLAP" else "dispatch_or_transition"
            )
            decisions.append(
                decision(
                    detector_id,
                    status="evaluated",
                    caught=bool(projected_details[detector_id]),
                    evidence_mode="projected_from_suppressed_event",
                    location={"stage": stage, "scope": location_scope},
                    source_paths=[raw_path],
                    detail={
                        "expected_gate": expected_gate,
                        "projected_events": projected_details[detector_id],
                        "independent_execution": False,
                    },
                )
            )
        else:
            decisions.append(unavailable(detector_id, "not_applicable", [raw_path], "no frozen conventional rule for this target in this case"))

    persistent_results: list[bool] = []
    for probe in outcome.get("probe_receipts", []):
        replay = probe.get("replay_receipt") or probe.get("receipt") or {}
        if "persistent_binding_and_digest_immutable" in replay:
            persistent_results.append(bool(replay["persistent_binding_and_digest_immutable"]))
        guard = (probe.get("storage_witness") or {}).get("persistent_guard") or {}
        if guard:
            persistent_results.append(
                guard.get("baseline_content_sha256") == guard.get("observed_content_sha256")
            )
    if persistent_results:
        decisions.append(
            decision(
                "B5_PERSISTENT_BASE_IMMUTABILITY",
                status="evaluated",
                caught=not all(persistent_results),
                evidence_mode="computed_from_raw_receipt",
                location={"stage": "post_transition_state", "scope": "persistent base digest"},
                source_paths=[raw_path],
                detail={"component_checks": persistent_results},
            )
        )
    else:
        decisions.append(unavailable("B5_PERSISTENT_BASE_IMMUTABILITY", "not_evaluated", [raw_path], "no persistent-base digest observer in this case"))
    return decisions


def load_primary_rows(repo_root: Path, tracker: EvidenceTracker, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    paper = repo_root / "paper_autonomous_multifork_iteration"
    r28 = paper / "evidence/r28_full_detector_matrix/formal_run_20260824a"
    summary_path = r28 / "detector-matrix-v2-summary.postexec-corrected.json"
    summary = tracker.load_json(summary_path, "primary corrected target-suppression summary")
    if not summary.get("scientific_valid") or summary.get("operational_invalid_count") != 0:
        raise ValueError("primary target-suppression summary is not scientifically valid")

    raw_cases: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    for path in sorted((r28 / "raw").glob("detector-matrix-v2-rank-*.json")):
        data = tracker.load_json(path, "primary target-suppression raw rank")
        relative = tracker.relative(path)
        for case in data["cases"]:
            key = (case["mutant_id"], case["lane"])
            if key in raw_cases:
                raise ValueError(f"duplicate primary case {key}")
            raw_cases[key] = (case, relative)

    reference_root = paper / "evidence/round_04_rr2_package/upstream/raw"
    rows = []
    summary_rows = summary["per_fault_detector_rows"]
    if [item["mutant_id"] for item in summary_rows] != [f"M{i}" for i in range(1, 10)]:
        raise ValueError("primary fault IDs/order drift")

    for item in summary_rows:
        mutant_id = item["mutant_id"]
        reference = item["rr2_forkaudit_reference"]
        shard_path = reference_root / reference["shard_relative_path"]
        shard = tracker.load_json(shard_path, "primary RR2 all-gates-on shard")
        if sha256_file(shard_path) != reference["shard_sha256"]:
            raise ValueError(f"RR2 shard hash mismatch for {mutant_id}")
        archived = shard["fault_campaign"]["mutants"][mutant_id]
        if archived["outcome"]["classification"] != "detected_expected_gate":
            raise ValueError(f"RR2 mutant outcome drift for {mutant_id}")
        if archived["outcome"]["observed_gate_id"] != reference["observed_gate_id"]:
            raise ValueError(f"RR2 observed gate drift for {mutant_id}")
        if archived["matched_clean"]["outcome"]["classification"] != "clean_pass":
            raise ValueError(f"RR2 matched clean failed for {mutant_id}")
        fork_sources = [tracker.relative(summary_path), tracker.relative(shard_path)]

        for lane, case_kind in (("target_suppressed", "fault"), ("clean", "clean")):
            case, raw_relative = raw_cases[(mutant_id, lane)]
            if case["outcome"]["valid_scientific_outcome"] is not True:
                raise ValueError(f"invalid primary case {mutant_id}/{lane}")
            is_fault = case_kind == "fault"
            rows.append(
                finalize_row(
                    campaign="primary_m1_m9",
                    case_id=f"primary/{mutant_id}/{case_kind}",
                    case_kind=case_kind,
                    fault_id=mutant_id,
                    coordinate=None,
                    baseline_decisions=primary_baseline_decisions(case, raw_relative, protocol),
                    forkaudit_detected=is_fault,
                    forkaudit_first_predicate=reference["observed_gate_id"] if is_fault else None,
                    forkaudit_location=primary_forkaudit_location(reference["observed_gate_id"]) if is_fault else None,
                    forkaudit_source_paths=fork_sources,
                    execution_relationship=(
                        "separate executions: conventional observers/projected checks use R28 target-suppression; "
                        "ForkAudit uses RR2 all-gates-on"
                    ),
                    protocol=protocol,
                )
            )
    return rows


def ranges_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("storage_id") != right.get("storage_id"):
        return False
    return max(int(left["byte_start"]), int(right["byte_start"])) < min(
        int(left["byte_end_exclusive"]), int(right["byte_end_exclusive"])
    )


def simple_cross_owner_overlaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlaps = []
    for left_index, left in enumerate(rows):
        left_owner = (left.get("owner_kind"), left.get("request_index"))
        for right in rows[left_index + 1 :]:
            right_owner = (right.get("owner_kind"), right.get("request_index"))
            if left_owner == right_owner:
                continue
            relevant = (
                {left.get("owner_kind"), right.get("owner_kind")} == {"persistent", "request"}
                or (
                    left.get("owner_kind") == right.get("owner_kind") == "request"
                    and left.get("request_index") != right.get("request_index")
                )
            )
            if relevant and ranges_overlap(left, right):
                overlaps.append(
                    {
                        "left": {
                            "owner_kind": left.get("owner_kind"),
                            "request_index": left.get("request_index"),
                            "layer_index": left.get("layer_index"),
                            "state_family": left.get("state_family"),
                            "state_index": left.get("state_index"),
                        },
                        "right": {
                            "owner_kind": right.get("owner_kind"),
                            "request_index": right.get("request_index"),
                            "layer_index": right.get("layer_index"),
                            "state_family": right.get("state_family"),
                            "state_index": right.get("state_index"),
                        },
                        "storage_id": left.get("storage_id"),
                    }
                )
    return overlaps


def source_digest_changes(case: dict[str, Any]) -> list[str]:
    physical = case.get("source_physical_digests") or {}
    setup = physical.get("setup") or {}
    transition = physical.get("transition") or {}
    final = physical.get("final") or {}
    layers = sorted(set(setup) | set(transition) | set(final), key=lambda value: int(value))
    return [layer for layer in layers if not (setup.get(layer) == transition.get(layer) == final.get(layer))]


def lifecycle_evidence(case: dict[str, Any]) -> tuple[bool | None, dict[str, Any]]:
    evidence = case.get("fault_specific_evidence") or {}
    if "ordered_tail_events" in evidence:
        events = evidence["ordered_tail_events"]
        copies = [event["ordinal"] for event in events if event["kind"] == "tail_copy"]
        writes = [event["ordinal"] for event in events if event["kind"] == "append_write"]
        premature = [event for event in events if event.get("premature_shared") is True]
        caught = bool(premature) or (bool(copies) and bool(writes) and min(writes) < min(copies))
        return caught, {"rule": "tail_copy_before_first_append", "events": events}
    if "extra_committed_call_count" in evidence:
        count = int(evidence["extra_committed_call_count"])
        return count != 0, {"rule": "extra_committed_call_count_equals_zero", "observed": count}
    if "stale_binding_token_count" in evidence:
        count = int(evidence["stale_binding_token_count"])
        return count != 0, {
            "rule": "stale_binding_token_count_equals_zero",
            "observed": count,
            "request_index": evidence.get("stale_request_index"),
        }
    return None, {"rule": None}


def r33_baseline_decisions(case: dict[str, Any], case_kind: str, source_path: str) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    decisions.append(unavailable("B1_RUNTIME_ASSERTION", "not_evaluated", [source_path], "no independent production assertion observer in R33 pair"))

    comparison = case.get("semantic_comparison_to_clean") if case_kind == "fault" else None
    if comparison is None:
        decisions.extend(
            [
                unavailable("B2_TOKEN_EQUALITY", "not_evaluated", [source_path], "clean case has no external semantic comparator"),
                unavailable("B3_FULL_LOGIT_EQUALITY", "not_evaluated", [source_path], "clean case has no external semantic comparator"),
            ]
        )
    else:
        decisions.append(
            decision(
                "B2_TOKEN_EQUALITY",
                status="evaluated",
                caught=not comparison["generated_tokens_exact"],
                evidence_mode="executed_independent_observer",
                location={"stage": "terminal_token", "scope": "generated token sequence"},
                source_paths=[source_path],
                detail={"generated_tokens_exact": comparison["generated_tokens_exact"]},
            )
        )
        if comparison["call_cardinality_comparable"]:
            decisions.append(
                decision(
                    "B3_FULL_LOGIT_EQUALITY",
                    status="evaluated",
                    caught=not comparison["full_fp32_logits_byte_exact"],
                    evidence_mode="executed_independent_observer",
                    location={"stage": "terminal_logit", "scope": "full FP32 logits"},
                    source_paths=[source_path],
                    detail={
                        "call_cardinality_comparable": True,
                        "full_fp32_logits_byte_exact": comparison["full_fp32_logits_byte_exact"],
                    },
                )
            )
        else:
            decisions.append(unavailable("B3_FULL_LOGIT_EQUALITY", "not_comparable", [source_path], "call cardinality differs"))

    target_call = (case.get("fault_specific_evidence") or {}).get("target_call")
    if target_call is not None:
        caught = target_call["observed_scale_hex"] != target_call["frozen_scale_hex"]
        decisions.append(
            decision(
                "B4_STRUCTURAL_SEQUENCE_ARGUMENT",
                status="evaluated",
                caught=caught,
                evidence_mode="computed_from_raw_receipt",
                location={"stage": "dispatch_or_transition", "scope": f"layer {target_call['layer_index']} attention scale"},
                source_paths=[source_path],
                detail=target_call,
            )
        )
    else:
        decisions.append(unavailable("B4_STRUCTURAL_SEQUENCE_ARGUMENT", "not_applicable", [source_path], "no frozen structural/scalar field for this R33 case"))

    persistent = case["existing_validator_receipts"]["persistent_gdn"]
    persistent_equal = (
        persistent["baseline_binding_sha256"] == persistent["observed_binding_sha256"]
        and persistent["baseline_content_sha256"] == persistent["observed_content_sha256"]
    )
    changed_layers = source_digest_changes(case)
    decisions.append(
        decision(
            "B5_PERSISTENT_BASE_IMMUTABILITY",
            status="evaluated",
            caught=(not persistent_equal) or bool(changed_layers),
            evidence_mode="computed_from_raw_receipt",
            location={"stage": "post_transition_state", "scope": "persistent GDN/document physical digest"},
            source_paths=[source_path],
            detail={"persistent_gdn_equal": persistent_equal, "changed_document_layers": changed_layers},
        )
    )

    storage_rows = case["gdn_storage_witness"]["rows"]
    overlaps = simple_cross_owner_overlaps(storage_rows)
    decisions.append(
        decision(
            "B6_SIMPLE_ALIAS_OVERLAP",
            status="evaluated",
            caught=bool(overlaps),
            evidence_mode="computed_from_raw_receipt",
            location={"stage": "post_transition_state", "scope": "request/base or request/peer storage interval"},
            source_paths=[source_path],
            detail={"overlap_count": len(overlaps), "overlaps": overlaps},
        )
    )

    lifecycle_caught, lifecycle_detail = lifecycle_evidence(case)
    if lifecycle_caught is None:
        decisions.append(unavailable("B7_BASIC_LIFECYCLE_CARDINALITY", "not_applicable", [source_path], "no lifecycle/cardinality field for this case"))
    else:
        decisions.append(
            decision(
                "B7_BASIC_LIFECYCLE_CARDINALITY",
                status="evaluated",
                caught=lifecycle_caught,
                evidence_mode="computed_from_raw_receipt",
                location={"stage": "dispatch_or_transition", "scope": lifecycle_detail["rule"]},
                source_paths=[source_path],
                detail=lifecycle_detail,
            )
        )
    return decisions


def r33_forkaudit_location(predicate: str | None) -> dict[str, Any] | None:
    if predicate is None:
        return None
    stage = {
        "TAIL_COPY_BEFORE_FIRST_APPEND_WRITE": "dispatch_or_transition",
        "PHYSICAL_DOCUMENT_PREFIX_IMMUTABLE": "post_transition_state",
        "ORDERED_CALL_CARDINALITY": "post_transition_state",
        "ATTENTION_EFFECTIVE_SCALE": "dispatch_or_transition",
        "GDN_COMPLETED_BINDING_TOKEN_ADVANCE": "post_transition_state",
    }[predicate]
    return {"stage": stage, "scope": predicate}


def load_r33_rows(repo_root: Path, tracker: EvidenceTracker, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    root = repo_root / "paper_autonomous_multifork_iteration/evidence/r33_fresh_faults/formal_h20/r33-fresh-faults-20260825b"
    summary = tracker.load_json(root / "summary.json", "R33 designer-executor aggregate")
    verification = tracker.load_json(root.parent / "RESULT_VERIFICATION.json", "R33 result verification")
    if summary.get("status") not in {"complete", "completed", "scientifically_valid"} and not summary.get("scientific_valid", False):
        # The exact aggregate uses a campaign-specific status; verification is authoritative below.
        if verification.get("verified") is not True and verification.get("all_checks_passed") is not True:
            raise ValueError("R33 aggregate/verification not valid")

    rows = []
    rank_dirs = sorted(root.glob("rank-run-*/rank-*"))
    if len(rank_dirs) != 5:
        raise ValueError(f"expected 5 R33 rank directories, found {len(rank_dirs)}")
    for rank_dir in rank_dirs:
        pair_path = rank_dir / "pair.json"
        clean_path = rank_dir / "clean-case.json"
        mutant_path = rank_dir / "mutant-case.json"
        clean_replay_path = rank_dir / "clean-gate-replay.json"
        pair_replay_path = rank_dir / "pair-replay.json"
        pair = tracker.load_json(pair_path, "R33 pair classification")
        clean = tracker.load_json(clean_path, "R33 clean raw case")
        mutant = tracker.load_json(mutant_path, "R33 mutant raw case")
        clean_replay = tracker.load_json(clean_replay_path, "R33 detached clean replay")
        pair_replay = tracker.load_json(pair_replay_path, "R33 detached pair replay")
        fault_id = pair["fault_id"]
        if pair["classification"] != "caught_by_expected_primary_gate":
            raise ValueError(f"R33 fault not caught as frozen: {fault_id}")
        if clean_replay["status"] != "clean_gate_passed":
            raise ValueError(f"R33 clean gate failed: {fault_id}")
        if pair_replay["classification"] != "caught_by_expected_primary_gate":
            raise ValueError(f"R33 detached replay mismatch: {fault_id}")
        pair_relative = tracker.relative(pair_path)
        replay_relative = tracker.relative(pair_replay_path)
        forkaudit_sources = [pair_relative, replay_relative]
        for case_kind, case, path in (
            ("fault", mutant, mutant_path),
            ("clean", clean, clean_path),
        ):
            relative = tracker.relative(path)
            is_fault = case_kind == "fault"
            rows.append(
                finalize_row(
                    campaign="designer_executor_r33",
                    case_id=f"designer/{fault_id}/{case_kind}",
                    case_kind=case_kind,
                    fault_id=fault_id,
                    coordinate=f"rank-{pair['rank']}",
                    baseline_decisions=r33_baseline_decisions(case, case_kind, relative),
                    forkaudit_detected=is_fault,
                    forkaudit_first_predicate=pair["first_failed_predicate"] if is_fault else None,
                    forkaudit_location=r33_forkaudit_location(pair["first_failed_predicate"]) if is_fault else None,
                    forkaudit_source_paths=forkaudit_sources + ([tracker.relative(clean_replay_path)] if not is_fault else []),
                    execution_relationship="same R33 case/capture; plain receipt checks are not independently instrumented except semantic observers",
                    protocol=protocol,
                )
            )
    return rows


def historical_baseline_decisions(
    lane: dict[str, Any],
    comparison: dict[str, Any],
    source_path: str,
) -> list[dict[str, Any]]:
    decisions = [unavailable("B1_RUNTIME_ASSERTION", "not_evaluated", [source_path], "authenticated ForkAudit rejection is not relabeled as a conventional runtime assertion")]
    decisions.append(
        decision(
            "B2_TOKEN_EQUALITY",
            status="evaluated",
            caught=not comparison["greedy_token_exact"],
            evidence_mode="executed_independent_observer",
            location={"stage": "terminal_token", "scope": "greedy token versus materialized control"},
            source_paths=[source_path],
            detail={"greedy_token_exact": comparison["greedy_token_exact"]},
        )
    )
    decisions.append(
        decision(
            "B3_FULL_LOGIT_EQUALITY",
            status="evaluated",
            caught=not comparison["full_fp32_logits_exact"],
            evidence_mode="executed_independent_observer",
            location={"stage": "terminal_logit", "scope": "full FP32 logits versus materialized control"},
            source_paths=[source_path],
            detail={"full_fp32_logits_exact": comparison["full_fp32_logits_exact"]},
        )
    )
    decisions.append(unavailable("B4_STRUCTURAL_SEQUENCE_ARGUMENT", "not_applicable", [source_path], "historical mechanism is not a structural/sequence argument fault"))

    guard = lane["persistent_guard_result"]
    guard_caught = guard["status"] == "authenticated_rejection"
    decisions.append(
        decision(
            "B5_PERSISTENT_BASE_IMMUTABILITY",
            status="evaluated",
            caught=guard_caught,
            evidence_mode="executed_independent_observer",
            location={"stage": "post_transition_state", "scope": "persistent-base content guard"},
            source_paths=[source_path],
            detail={"guard_status": guard["status"], "guard": guard},
        )
    )

    relation_rows = lane["post_ownership_relations"]["rows"]
    overlaps = []
    for row in relation_rows:
        for relation_name in ("request_0_vs_base", "request_0_vs_peer"):
            relation = row[relation_name]
            if relation["ranges_overlap"] or relation["exact_byte_range_alias"]:
                overlaps.append(
                    {
                        "coordinate": row["coordinate"],
                        "state_family": row["state_family"],
                        "relation": relation_name,
                        "same_storage": relation["same_storage"],
                        "ranges_overlap": relation["ranges_overlap"],
                        "exact_byte_range_alias": relation["exact_byte_range_alias"],
                    }
                )
    decisions.append(
        decision(
            "B6_SIMPLE_ALIAS_OVERLAP",
            status="evaluated",
            caught=bool(overlaps),
            evidence_mode="computed_from_raw_receipt",
            location={"stage": "post_transition_state", "scope": "completed request versus base/peer intervals"},
            source_paths=[source_path],
            detail={"overlap_count": len(overlaps), "overlaps": overlaps},
        )
    )
    decisions.append(unavailable("B7_BASIC_LIFECYCLE_CARDINALITY", "not_applicable", [source_path], "historical mechanism is an ownership alias, not lifecycle/cardinality"))
    return decisions


def load_historical_rows(repo_root: Path, tracker: EvidenceTracker, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    root = repo_root / "paper_autonomous_multifork_iteration/evidence/r35_historical_alias_regression/formal_h20/r35-historical-alias-20260826a"
    aggregate = tracker.load_json(root / "aggregate.json", "R35 historical aggregate")
    verification = tracker.load_json(root.parent / "RESULT_VERIFICATION.json", "R35 result verification")
    if not aggregate["operational_valid"] or aggregate["status"] != "operationally_valid_aggregate":
        raise ValueError("R35 aggregate invalid")
    if verification.get("verified") is False or verification.get("all_checks_passed") is False:
        raise ValueError("R35 verification failed")
    rows = []
    for rank in range(8):
        result_path = root / f"rank-{rank}/raw/rank-result.json"
        replay_path = root / f"replay/rank-{rank}-replay.json"
        result = tracker.load_json(result_path, "R35 raw rank result")
        replay = tracker.load_json(replay_path, "R35 detached rank replay")
        if result["operational_invalid"] is True or result["status"] != "rank_completed":
            raise ValueError(f"R35 rank {rank} invalid")
        if replay.get("status") not in {"replay_passed", "passed", "complete", "completed"}:
            # Some versions expose only exact booleans; require no explicit failure.
            if any(value is False for key, value in replay.items() if key.endswith("_passed")):
                raise ValueError(f"R35 replay {rank} failed")
        source_paths = [tracker.relative(result_path), tracker.relative(replay_path)]
        comparisons = result["comparisons"]
        for lane_name in ("historical_pre_fix", "repaired_borrowed", "materialized_control"):
            lane = result["lanes"][lane_name]
            is_fault = lane_name == "historical_pre_fix"
            if lane_name == "historical_pre_fix":
                comparison = comparisons["historical_pre_fix_vs_materialized_control"]
            else:
                comparison = comparisons["repaired_borrowed_vs_materialized_control"]
            first = lane["audit"].get("first_authenticated_rejection")
            if is_fault:
                if lane["status"] != "authenticated_forkaudit_rejection_after_model_step" or not first:
                    raise ValueError(f"R35 historical lane {rank} did not reproduce rejection")
            else:
                if lane["status"] != "completed_clean" or first is not None:
                    raise ValueError(f"R35 clean lane {rank}/{lane_name} failed")
            coordinate_class = "archived" if rank < 3 else "additional_frozen_input"
            rows.append(
                finalize_row(
                    campaign="historical_alias_r35",
                    case_id=f"historical/rank-{rank}/{lane_name}",
                    case_kind="fault" if is_fault else "clean",
                    fault_id="HISTORICAL_GDN_CONV_ALIAS" if is_fault else None,
                    coordinate=f"{coordinate_class}/rank-{rank}",
                    baseline_decisions=historical_baseline_decisions(lane, comparison, tracker.relative(result_path)),
                    forkaudit_detected=is_fault,
                    forkaudit_first_predicate=first["predicate_id"] if first else None,
                    forkaudit_location=(
                        {
                            "stage": "post_transition_state",
                            "scope": f"{first['receipt_id']} / {first['predicate_id']}",
                        }
                        if first
                        else None
                    ),
                    forkaudit_source_paths=source_paths,
                    execution_relationship="same R35 lane/capture; output and persistent-base observers are explicit baseline comparisons; alias check is recomputed",
                    protocol=protocol,
                )
            )
    return rows


def summarize(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    campaigns = sorted({row["campaign"] for row in rows})

    def summary_for(selected: list[dict[str, Any]]) -> dict[str, Any]:
        faults = [row for row in selected if row["case_kind"] == "fault"]
        cleans = [row for row in selected if row["case_kind"] == "clean"]
        relations = Counter(row["catch_relation"] for row in faults)
        return {
            "fault_case_count": len(faults),
            "clean_case_count": len(cleans),
            "baseline_caught_fault_case_count": sum(row["baseline"]["detected"] for row in faults),
            "strict_independent_baseline_caught_fault_case_count": sum(
                row["baseline"]["strict_independent_baseline_detected"] for row in faults
            ),
            "forkaudit_caught_fault_case_count": sum(row["forkaudit"]["detected"] for row in faults),
            "forkaudit_unique_fault_case_count": relations["forkaudit_unique"],
            "redundant_both_fault_case_count": relations["redundant_both"],
            "baseline_only_fault_case_count": relations["baseline_only"],
            "neither_fault_case_count": relations["neither"],
            "baseline_clean_false_positive_case_count": sum(row["baseline"]["detected"] for row in cleans),
            "forkaudit_clean_false_positive_case_count": sum(row["forkaudit"]["detected"] for row in cleans),
        }

    detector_fault_catches = Counter()
    detector_clean_catches = Counter()
    evidence_modes = Counter()
    for row in rows:
        for detector_id, item in row["baseline"]["detectors"].items():
            evidence_modes[item["evidence_mode"]] += 1
            if item["caught"] is True:
                if row["case_kind"] == "fault":
                    detector_fault_catches[detector_id] += 1
                else:
                    detector_clean_catches[detector_id] += 1

    return {
        "schema_version": "forkaudit-r40-baseline-summary-v1",
        "analysis_id": ANALYSIS_ID,
        "scientific_status": "retrospective_archived_evidence_analysis_complete",
        "case_counts_only_not_population_rates": True,
        "overall": summary_for(rows),
        "by_campaign": {
            campaign: summary_for([row for row in rows if row["campaign"] == campaign])
            for campaign in campaigns
        },
        "baseline_detector_fault_catch_counts": {
            detector_id: detector_fault_catches[detector_id] for detector_id in DETECTOR_IDS
        },
        "baseline_detector_clean_false_positive_counts": {
            detector_id: detector_clean_catches[detector_id] for detector_id in DETECTOR_IDS
        },
        "detector_decision_evidence_mode_counts": dict(sorted(evidence_modes.items())),
        "claim_boundary": protocol["claim_boundary"],
        "interpretation": {
            "full_baseline": "includes executed observers, recomputed raw-receipt rules, and explicitly labeled primary suppressed-event projections",
            "strict_independent_subset": "includes only caught decisions labeled executed_independent_observer",
            "historical_rows": "eight coordinates of one defect mechanism, not eight defects",
            "false_positive_counts": "counts over these evaluated clean controls only; not a precision estimate",
        },
    }


def rows_to_csv(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    fieldnames = [
        "campaign",
        "case_id",
        "case_kind",
        "fault_id",
        "coordinate",
        "baseline_detected",
        "strict_independent_baseline_detected",
        "baseline_caught_detectors",
        "baseline_first_detector",
        "baseline_first_stage",
        "baseline_first_scope",
        "forkaudit_detected",
        "forkaudit_first_predicate",
        "forkaudit_first_stage",
        "forkaudit_first_scope",
        "catch_relation",
        "execution_relationship",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        baseline_first = row["baseline"]["first_localization"] or {}
        baseline_location = baseline_first.get("location") or {}
        fork_location = row["forkaudit"]["first_localization"] or {}
        writer.writerow(
            {
                "campaign": row["campaign"],
                "case_id": row["case_id"],
                "case_kind": row["case_kind"],
                "fault_id": row["fault_id"] or "",
                "coordinate": row["coordinate"] or "",
                "baseline_detected": str(row["baseline"]["detected"]).lower(),
                "strict_independent_baseline_detected": str(row["baseline"]["strict_independent_baseline_detected"]).lower(),
                "baseline_caught_detectors": ";".join(row["baseline"]["caught_detector_ids"]),
                "baseline_first_detector": baseline_first.get("detector_id", ""),
                "baseline_first_stage": baseline_location.get("stage", ""),
                "baseline_first_scope": baseline_location.get("scope", ""),
                "forkaudit_detected": str(row["forkaudit"]["detected"]).lower(),
                "forkaudit_first_predicate": row["forkaudit"]["first_failed_predicate"] or "",
                "forkaudit_first_stage": fork_location.get("stage", ""),
                "forkaudit_first_scope": fork_location.get("scope", ""),
                "catch_relation": row["catch_relation"],
                "execution_relationship": row["execution_relationship"],
            }
        )
    return buffer.getvalue().encode("utf-8")


def validate_cardinality(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    expected = protocol["case_cardinality"]
    observed = {
        "primary_fault": sum(row["campaign"] == "primary_m1_m9" and row["case_kind"] == "fault" for row in rows),
        "primary_clean": sum(row["campaign"] == "primary_m1_m9" and row["case_kind"] == "clean" for row in rows),
        "designer_executor_fault": sum(row["campaign"] == "designer_executor_r33" and row["case_kind"] == "fault" for row in rows),
        "designer_executor_clean": sum(row["campaign"] == "designer_executor_r33" and row["case_kind"] == "clean" for row in rows),
        "historical_defect_coordinate": sum(row["campaign"] == "historical_alias_r35" and row["case_kind"] == "fault" for row in rows),
        "historical_clean": sum(row["campaign"] == "historical_alias_r35" and row["case_kind"] == "clean" for row in rows),
    }
    if observed != expected:
        raise ValueError(f"case cardinality drift: expected={expected}, observed={observed}")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case IDs")


def build_products(repo_root: Path) -> dict[str, bytes]:
    repo_root = repo_root.resolve()
    analysis_root = Path(__file__).resolve().parent
    protocol_path = analysis_root / "protocol.json"
    tracker = EvidenceTracker(repo_root)
    protocol = tracker.load_json(protocol_path, "frozen R40 protocol")
    tracker.register(Path(__file__), "R40 matrix generator")
    tracker.register(analysis_root / "FROZEN_PROTOCOL.md", "human-readable frozen R40 protocol")
    if protocol["analysis_id"] != ANALYSIS_ID or protocol["outcome_blinded"] is not False:
        raise ValueError("protocol identity/disclosure drift")

    rows = []
    rows.extend(load_primary_rows(repo_root, tracker, protocol))
    rows.extend(load_r33_rows(repo_root, tracker, protocol))
    rows.extend(load_historical_rows(repo_root, tracker, protocol))
    rows.sort(key=lambda row: row["case_id"])
    validate_cardinality(rows, protocol)
    summary = summarize(rows, protocol)
    matrix = {
        "schema_version": "forkaudit-r40-baseline-detector-matrix-v1",
        "analysis_id": ANALYSIS_ID,
        "retrospective": True,
        "outcome_blinded": False,
        "case_counts_only_not_population_rates": True,
        "row_count": len(rows),
        "rows": rows,
    }
    manifest = tracker.manifest()

    products: dict[str, bytes] = {
        "baseline_detector_matrix.json": canonical_json_bytes(matrix),
        "baseline_detector_matrix.csv": rows_to_csv(rows),
        "summary.json": canonical_json_bytes(summary),
        "input_manifest.json": canonical_json_bytes(manifest),
    }
    product_hashes = {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(products.items())
    }
    receipt = {
        "schema_version": "forkaudit-r40-baseline-run-receipt-v1",
        "analysis_id": ANALYSIS_ID,
        "status": "complete",
        "command": "python3 build_matrix.py --repo-root <REPO_ROOT> --output-dir results",
        "generator_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(protocol_path),
        "input_manifest_sha256": product_hashes["input_manifest.json"]["sha256"],
        "products": product_hashes,
        "no_new_model_execution": True,
        "no_gpu_or_qs_access": True,
    }
    products["run_receipt.json"] = canonical_json_bytes(receipt)
    lines = [f"{sha256_bytes(products[name])}  {name}\n" for name in sorted(products)]
    products["SHA256SUMS"] = "".join(lines).encode("utf-8")
    return products


def materialize(products: dict[str, bytes], output_dir: Path, verify_existing: bool) -> None:
    if verify_existing:
        if not output_dir.is_dir():
            raise FileNotFoundError(output_dir)
        expected_names = set(products)
        actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
        if actual_names != expected_names:
            raise ValueError(f"output member drift: expected={sorted(expected_names)}, actual={sorted(actual_names)}")
        mismatches = [name for name, payload in products.items() if (output_dir / name).read_bytes() != payload]
        if mismatches:
            raise ValueError(f"deterministic replay mismatch: {mismatches}")
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in sorted(products.items()):
        (output_dir / name).write_bytes(payload)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    products = build_products(args.repo_root)
    materialize(products, args.output_dir.resolve(), args.verify_existing)
    summary = json.loads(products["summary.json"])
    print(json.dumps(summary["overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
