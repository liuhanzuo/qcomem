#!/usr/bin/env python3
"""Fail-closed local integration audit for the R41/V29 paper successor."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
TEX = PAPER / "main_r41_v29_live_binding.tex"
FROZEN_R40 = PAPER / "main_r40_submission_candidate.tex"
PDF = PAPER / "build/r41_v29_live_binding_final/main_r41_v29_live_binding.pdf"
LOG = PAPER / "build/r41_v29_live_binding_final/main_r41_v29_live_binding.log"
REGISTRY = PAPER / "evidence/experiment_registry.json"
CLAIMS = PAPER / "evidence/claim_evidence_map.tsv"
METHODS = PAPER / "evidence/method_provenance.tsv"
AUDIT = PAPER / "evidence/r40_independent_live_binding_v29_postrun_audit_mirror/POSTRUN_INDEPENDENT_AUDIT.json"
EVIDENCE_ID = "E-R40-INDEPENDENT-LIVE-BINDING-V29-A"
CLAIM_ID = "C-R40-INDEPENDENT-LIVE-BINDING-V29-01"
FROZEN_R40_SHA256 = "166dff9f56da4449a53857c575fbf9f62466c7bd1e84b7c05e59301ffe346c10"
AUDIT_SHA256 = "7af92c70ec48bd35ad44bf475ddbba556b65e6a7c9cff891674f9688a464450d"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path, columns: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(rows and len(rows[0]) == columns, f"unexpected TSV schema: {path}")
    require(all(None not in row for row in rows), f"malformed TSV row: {path}")
    return rows


def verify_includes(tex: str) -> int:
    checked = 0
    for kind, raw_path in re.findall(
        r"\\(input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", tex
    ):
        relative = Path(raw_path)
        candidates = [PAPER / relative]
        if kind == "input" and not relative.suffix:
            candidates.append(PAPER / f"{raw_path}.tex")
        if kind == "includegraphics" and not relative.suffix:
            candidates.extend(PAPER / f"{raw_path}{suffix}" for suffix in (".pdf", ".png", ".jpg"))
        require(any(path.is_file() for path in candidates), f"missing {kind}: {raw_path}")
        checked += 1
    return checked


def compact_page(page: int) -> str:
    output = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(PDF), "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    return re.sub(r"[^A-Z0-9]", "", output.upper())


def main() -> None:
    require(TEX.is_file() and PDF.is_file() and LOG.is_file(), "R41 source/final build missing")
    require(sha256_file(FROZEN_R40) == FROZEN_R40_SHA256, "frozen R40 candidate changed")
    require(sha256_file(AUDIT) == AUDIT_SHA256, "V29 independent audit mirror drift")
    tex = TEX.read_text(encoding="utf-8")
    lowered = tex.lower()

    forbidden = [
        "v26 as positive", "v27 as positive", "v28 as positive",
        "runtime-independent system correctness", "malicious-runtime resistance",
        "all cells are live-bound", "/users/", "liuhanzuo",
    ]
    require(not any(phrase in lowered for phrase in forbidden),
            "stale, identifying, or prohibited expansion in R41 source")
    required = [
        "all seven", "209,920", "635,520", "224", "184",
        "144 selected rows", "12,960", "3,840", "24 stable",
        "all 96 clean-memory calls", "zero selected-context overlaps",
        "six preregistered rows", "all 540 live rows", "honest same-process",
        "E-R40-INDEPENDENT-LIVE-BINDING-V29-A", "54.5\\%",
        "Reproducibility Statement",
    ]
    require(all(phrase in tex for phrase in required), "required V29 claim/boundary missing")
    repro = tex.index(r"\section*{Reproducibility Statement}")
    bibliography = tex.index(r"\bibliography{references}")
    appendix = tex.index(r"\appendix")
    require(repro < bibliography < appendix, "main/references/appendix order drift")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    experiments = registry["experiments"]
    ids = [row["evidence_id"] for row in experiments]
    require(len(ids) == len(set(ids)), "duplicate evidence ID")
    evidence = next(row for row in experiments if row["evidence_id"] == EVIDENCE_ID)
    validation = evidence["validation"]
    require(evidence["active_manuscript_support"] is True, "V29 not active manuscript support")
    require(evidence["authorization"]["manuscript_integrated"] is True, "V29 manuscript integration not closed")
    require(evidence["authorization"]["fresh_review_completed"] is True, "V29 fresh review not closed")
    expected_counts = {
        "rank_count": 8,
        "total_selected_rows": 144,
        "total_storage_rows": 12960,
        "total_direct_clone_edges": 3840,
        "total_phase_artifacts": 24,
        "total_primary_calls_observed": 96,
        "global_primary_memory_hook_events": 0,
        "terminal_final_node_count": 1367,
    }
    require(all(validation[key] == value for key, value in expected_counts.items()),
            "V29 registry count drift")
    require(validation["postrun_read_only_audit_passed"] is True, "post-run audit not authorized")
    for relative in evidence["artifact_paths"]:
        require((PAPER / relative).is_file(), f"missing V29 artifact: {relative}")

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    require(audit["audit_status"] == "pass" and audit["read_only_audit"] is True,
            "independent audit status drift")
    require(audit["primary"]["scientific_outcome"] == "valid_positive", "scientific outcome drift")
    require(audit["formal_dispatch"]["negative_controls_rejected"] == 184,
            "V29 formal control count drift")
    totals = audit["r40_live_binding"]["totals"]
    require(totals == {
        "global_primary_memory_hook_events": 0,
        "rank_count": 8,
        "total_clone_edges": 3840,
        "total_phase_artifacts": 24,
        "total_primary_calls_observed": 96,
        "total_selected_rows": 144,
        "total_storage_rows": 12960,
    }, "V29 audit totals drift")
    require(audit["terminal"]["final_node_count"] == 1367, "terminal tree count drift")

    claim_rows = read_tsv(CLAIMS, 7)
    claim_ids = [row["claim_id"] for row in claim_rows]
    require(len(claim_ids) == len(set(claim_ids)) and CLAIM_ID in claim_ids,
            "V29 claim registration missing/duplicated")
    claim = next(row for row in claim_rows if row["claim_id"] == CLAIM_ID)
    require(EVIDENCE_ID in claim["evidence_ids"], "V29 claim/evidence link drift")
    require("pending" not in claim["status_or_notes"].lower(), "V29 claim still marked pending")

    method_rows = read_tsv(METHODS, 6)
    method_ids = [row["method_id"] for row in method_rows]
    require(len(method_ids) == len(set(method_ids)), "duplicate method ID")
    for method_id in (
        "M-R40-LIVE-BINDING-V29-CAPTURE",
        "M-R40-LIVE-BINDING-V29-FINALIZE",
        "M-R40-LIVE-BINDING-V29-TERMINAL-CLOSURE",
    ):
        row = next(row for row in method_rows if row["method_id"] == method_id)
        require("pending" not in row["validation_boundary"].lower(),
                f"method provenance still pending: {method_id}")

    log = LOG.read_text(encoding="utf-8")
    require(not re.search(r"LaTeX Warning:|undefined references|Citation .* undefined|Reference .* undefined|Overfull", log),
            "final LaTeX log contains unresolved warning/overflow")
    page9 = compact_page(9)
    page10 = compact_page(10)
    require("CONCLUSION" in page9 and "REPRODUCIBILITYSTATEMENT" in page9,
            "nine-page main-text boundary not satisfied")
    require("REFERENCES" in page10 and "CONCLUSION" not in page10 and "REPRODUCIBILITYSTATEMENT" not in page10,
            "main text spills onto page 10")

    print(json.dumps({
        "status": "pass",
        "source_sha256": sha256_file(TEX),
        "pdf_sha256": sha256_file(PDF),
        "included_files_checked": verify_includes(tex),
        "registry_evidence_ids": len(ids),
        "claim_rows": len(claim_rows),
        "method_rows": len(method_rows),
        "v29_postrun_audit_sha256": sha256_file(AUDIT),
        "main_text_boundary": "conclusion_and_reproducibility_on_page_9_references_on_page_10",
        "gpu_work_executed_by_validator": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
