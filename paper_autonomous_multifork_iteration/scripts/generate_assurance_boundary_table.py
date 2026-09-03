#!/usr/bin/env python3
"""Generate the evidence-bound assurance summary and detector-comparison table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
MATCHED = PAPER / "evidence/round_04_rr2_package/derived/matched_pairs_9.json"
SCHEDULER = PAPER / "evidence/round23_a2_scheduler_interleave/formal_run_20260821a/scheduler-interleave-formal-summary.json"
PACKAGE_MANIFEST = PAPER / "evidence/round_04_rr2_package/MANIFEST.json"
RAW_LEDGER = PAPER / "evidence/round_04_rr2_package/derived/raw_ledger_validation.json"
R28_SUMMARY = PAPER / (
    "evidence/r28_full_detector_matrix/formal_run_20260824a/"
    "detector-matrix-v2-summary.postexec-corrected.json"
)
R28_REPLAY_RECEIPT = PAPER / (
    "evidence/r28_full_detector_matrix/formal_run_20260824a/"
    "postexecution-correction-replay-receipt.json"
)
OUTPUT_JSON = PAPER / "evidence/assurance_boundary_summary.json"
OUTPUT_TEX = PAPER / "tables/first_gate_localization_table.tex"

EXPECTED_SHA256 = {
    MATCHED: "a36484cc5179832e447e241e5707e7ee34d85ec84a02131c97adeb7150e2012b",
    SCHEDULER: "3fa46011ec65b921ffcf4f36b1294fc01d3c4d6b565e1645ee6da1c7d3600d21",
    PACKAGE_MANIFEST: "51346e18c2d2685ea57712d1823e6056ea6bea11a5718da6d24f2fe1d1b65338",
    RAW_LEDGER: "1ab1a803d59ae06db978760806f29dc9b5abdaf309f064dd3c766cd8010a6027",
    R28_SUMMARY: "94125f60bff8f3390716303c4f4d6262de894e62f932ba00f556ea9a3b97521e",
    R28_REPLAY_RECEIPT: "0e17e8848f170a3e973bdece79a93a50834b2147e513c3c2c3b71e52f53d21ea",
}
LIFECYCLE_STORAGE_GATES = {
    "KV_RESERVATION_DISJOINT",
    "KV_TAIL_COW",
    "gdn_completed_vs_base_disjoint",
    "gdn_completed_vs_peers_disjoint",
}
BINDING_CALL_GATES = {
    "KV_SEQUENCE_ID",
    "POSITION_CANONICAL_VALUES",
    "MASK_CONTRACT",
    "KERNEL_CALLABLE_ID",
    "KV_PAGED_VIEW",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path, expected in EXPECTED_SHA256.items():
        require(sha256(path) == expected, f"frozen source drift: {path}")

    matched = json.loads(MATCHED.read_text())
    scheduler = json.loads(SCHEDULER.read_text())
    package = json.loads(PACKAGE_MANIFEST.read_text())
    raw = json.loads(RAW_LEDGER.read_text())
    r28 = json.loads(R28_SUMMARY.read_text())
    r28_replay = json.loads(R28_REPLAY_RECEIPT.read_text())

    require(matched["matched_pair_count"] == 9, "matched-pair count drift")
    require(matched["all_matched_clean_passed"] is True, "matched clean failure")
    require(
        matched["all_mutants_detected_at_expected_gate"] is True,
        "expected-gate outcome drift",
    )
    rows = matched["rows"]
    require(len(rows) == 9, "mutant row coverage drift")
    require(
        all(row["detector_path_completed_after_gate"] is False for row in rows),
        "a mutant unexpectedly completed after its first gate",
    )
    require(
        all(
            row["detector_path_reached_expected_gate"] is True
            and row["expected_gate_id"] == row["observed_gate_id"]
            and row["matched_clean_classification"] == "clean_pass"
            for row in rows
        ),
        "matched first-gate classification drift",
    )

    lifecycle_rows = [
        row for row in rows if row["expected_gate_id"] in LIFECYCLE_STORAGE_GATES
    ]
    binding_rows = [
        row for row in rows if row["expected_gate_id"] in BINDING_CALL_GATES
    ]
    require(
        {row["mutant_id"] for row in lifecycle_rows} == {"M1", "M3", "M4", "M5"},
        "lifecycle/storage fault partition drift",
    )
    require(
        {row["mutant_id"] for row in binding_rows} == {"M2", "M6", "M7", "M8", "M9"},
        "binding/call fault partition drift",
    )
    require(len(lifecycle_rows) + len(binding_rows) == 9, "fault partition incomplete")

    require(scheduler["scientific_run_valid"] is True, "scheduler validity drift")
    require(
        scheduler["all_clean_semantic_and_storage_gates_passed"] is True,
        "scheduler clean-control drift",
    )
    require(scheduler["rank_count"] * scheduler["geometry_count_per_rank"] == 16,
            "scheduler clean-cell count drift")
    require(scheduler["heldout_fault_trial_count"] == 48,
            "scheduler fault-trial count drift")
    require(scheduler["heldout_fault_expected_gate_misses"] == 0,
            "scheduler expected-gate miss drift")

    require(package["file_count"] == 628, "package file-count drift")
    require(package["total_bytes"] == 892_144_066, "package byte-count drift")
    require(raw["artifact_count"] == 536, "raw artifact-count drift")
    require(raw["total_bytes"] == 888_785_811, "raw byte-count drift")
    require(raw["all_sha256_verified"] is True, "raw SHA verification drift")

    require(r28["scientific_valid"] is True, "R28 scientific validity drift")
    require(r28["scientific_outcome"] == "mixed", "R28 outcome drift")
    require(
        r28["counts"]
        == {
            "ranks": 8,
            "distinct_h20_gpu_uuids": 8,
            "cases": 18,
            "clean_cases": 9,
            "target_suppressed_mutant_cases": 9,
            "clean_fp32_sidecars": 14,
            "mutant_fp32_sidecars": 10,
            "measured_non_forkaudit_escapes": 4,
            "classifications": {
                "completed_semantics": 5,
                "other_forkaudit_gate": 2,
                "production_assertion": 1,
                "fault_payload_abort": 1,
                "operational_invalid": 0,
            },
        },
        "R28 count drift",
    )
    require(
        r28_replay["byte_identical_to_recorded_summary"] is True
        and r28_replay["summary_sha256"] == EXPECTED_SHA256[R28_SUMMARY]
        and r28_replay["scientific_valid"] is True,
        "R28 corrected replay drift",
    )
    r28_rows = {row["mutant_id"]: row for row in r28["per_fault_detector_rows"]}
    require(set(r28_rows) == {f"M{index}" for index in range(1, 10)}, "R28 row coverage")
    completed = {"M1", "M2", "M5", "M6", "M7"}
    for mutant_id in completed:
        outcome = r28_rows[mutant_id]["r28_target_suppressed"]
        require(outcome["classification"] == "completed_semantics", f"{mutant_id} class")
        require(outcome["token_only"]["caught"] is False, f"{mutant_id} token result")
    require(
        r28_rows["M5"]["r28_target_suppressed"]["full_logit"]["caught"] is True,
        "M5 full-logit result",
    )
    for mutant_id in {"M1", "M2", "M6", "M7"}:
        require(
            r28_rows[mutant_id]["r28_target_suppressed"]["full_logit"]["caught"]
            is False,
            f"{mutant_id} full-logit result",
        )
    require(
        r28_rows["M3"]["r28_target_suppressed"]["classification"]
        == r28_rows["M4"]["r28_target_suppressed"]["classification"]
        == "other_forkaudit_gate",
        "R28 redundant-gate result",
    )
    require(
        r28_rows["M8"]["r28_target_suppressed"]["classification"]
        == "fault_payload_abort"
        and r28_rows["M9"]["r28_target_suppressed"]["classification"]
        == "production_assertion",
        "R28 terminal classifications",
    )

    summary = {
        "schema_version": "forkaudit-assurance-boundary-derived-v1",
        "source_sha256": {str(path.relative_to(PAPER)): digest for path, digest in EXPECTED_SHA256.items()},
        "primary_first_gate_localization": {
            "matched_clean_controls_passed": 9,
            "mutants_rejected_at_predeclared_first_gate": 9,
            "post_injection_output_or_semantic_digest_available": 0,
            "lifecycle_storage_fault_ids": ["M1", "M3", "M4", "M5"],
            "binding_call_fault_ids": ["M2", "M6", "M7", "M8", "M9"],
            "interpretation": "first-gate localization only; output and semantic-digest checks are not observed rather than misses",
        },
        "scheduler_extension": {
            "clean_rank_geometry_cells_passed": 16,
            "preregistered_fault_trials_at_expected_gate": 48,
            "expected_gate_misses": 0,
            "interpretation": "same-stack deterministic request-step interleaving, not concurrent CUDA kernels or detector completeness",
        },
        "target_gate_suppression_matrix": {
            "cases": 18,
            "matched_clean_controls_passed": 9,
            "target_suppressed_mutants": 9,
            "completed_semantic_paths": 5,
            "token_only_catches_among_completed": 0,
            "full_logit_catches_among_completed": 1,
            "other_forkaudit_gate_catches": 2,
            "production_assertion_catches": 1,
            "fault_payload_aborts": 1,
            "measured_non_forkaudit_escapes": 4,
            "operational_invalid": 0,
            "interpretation": (
                "prospective per-fault same-system comparison; the injected M8 "
                "sentinel is not a production detector and no row is pooled into a rate"
            ),
        },
        "artifact_footprint": {
            "source_complete_files": 628,
            "source_complete_bytes": 892_144_066,
            "source_complete_mib": 892_144_066 / 2**20,
            "raw_bound_artifacts": 536,
            "raw_bound_bytes": 888_785_811,
            "raw_bound_mib": 888_785_811 / 2**20,
            "capture_time_measured": False,
            "complete_replay_time_measured": False,
            "interpretation": "uncompressed archival bytes, not online latency, device memory, traffic, or adoption cost",
        },
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    tex = r"""\begin{table}[H]
\caption{Same-system outcomes for nine designed faults.  The primary campaign
enables all gates; a separate target-gate-suppression campaign suppresses only
the named gate.
``Same'' is exact and ``N/O'' an earlier stop.  Rows are not a detection rate;
M8's injected sentinel is not a production detector.}
\label{tab:first-gate-localization}
\centering\scriptsize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}p{0.19\linewidth}p{0.12\linewidth}p{0.39\linewidth}p{0.18\linewidth}@{}}
\toprule
Fault / target contract & All gates on & With target gate suppressed & Token / FP32 logits \\
\midrule
M1 Reservation ownership & target gate & completes & Same / Same \\
M2 Sequence binding & target gate & completes & Same / Same \\
M3 Tail COW & target gate & other audit: active-block ownership & N/O / N/O \\
M4 GDN base isolation & target gate & other audit: peer isolation & N/O / N/O \\
M5 GDN peer isolation & target gate & completes; logits change ($L_\infty=0.75$, rel. $L_2=0.0314$) & Same / Changed \\
M6 Positions & target gate & completes & Same / Same \\
M7 Mask contract & target gate & completes & Same / Same \\
M8 Callable identity & target gate & injected payload sentinel & N/O / N/O \\
M9 Paged-KV view & target gate & paired-view production assertion & N/O / N/O \\
\bottomrule
\end{tabular}
\end{table}
"""
    OUTPUT_TEX.write_text(tex)
    print(json.dumps({"summary": str(OUTPUT_JSON), "table": str(OUTPUT_TEX)}))


if __name__ == "__main__":
    main()
