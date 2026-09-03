#!/usr/bin/env python3
"""Static guard for the Round-4 Appendix-F scope repair.

This audit is deliberately text-level: it prevents the manuscript from silently recovering
the retired allocation-wide/reselected conclusion without a new uniform proof. It does not
evaluate the frozen experiment artifacts or establish the FIT/CAL application contract.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
TEX = PROJECT / "manuscript" / "paper.tex"
CLAIMS = PROJECT / "evidence" / "claim_evidence_map.tsv"
PROVENANCE = PROJECT / "evidence" / "method_provenance.tsv"
FIGURE = PROJECT / "manuscript" / "fig_m10_frontier.pdf"
OUTPUT = PROJECT / "build" / "revision_04_allocation_scope_static_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def between(text: str, start: str, end: str) -> str:
    left = text.find(start)
    right = text.find(end, left + len(start))
    if left < 0 or right < 0:
        raise ValueError(f"cannot delimit {start!r} ... {end!r}")
    return text[left:right]


def require_all(text: str, needles: list[str], label: str, checks: list[dict]) -> None:
    normalized_text = " ".join(text.split())
    for needle in needles:
        checks.append(
            {
                "check": f"{label}: requires {needle}",
                "passed": " ".join(needle.split()) in normalized_text,
            }
        )


def reject_all(text: str, needles: list[str], label: str, checks: list[dict]) -> None:
    normalized_text = " ".join(text.split())
    for needle in needles:
        checks.append(
            {
                "check": f"{label}: rejects {needle}",
                "passed": " ".join(needle.split()) not in normalized_text,
            }
        )


def main() -> int:
    tex = TEX.read_text()
    claims = CLAIMS.read_text()
    provenance = PROVENANCE.read_text()
    figure_text = subprocess.run(
        ["pdftotext", str(FIGURE), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    checks: list[dict] = []

    m6 = between(tex, r"\paragraph{$M6$", r"\paragraph{$M7$")
    m9_to_m12 = between(tex, r"\paragraph{$M9$/$M10$", r"\paragraph{$M12$")
    caption = between(tex, r"\caption{\textbf{Fixed-statistic", r"\label{fig:m10frontier}")
    m12 = between(tex, r"\paragraph{$M12$", r"\section{AI use statement}")

    require_all(
        m6,
        [
            "asymptotic/descriptive relative-gate diagnostic",
            "not a finite-sample certificate or a safe-upgrade finding",
            "only keeps the status quo",
            "absolute $M2.5$ gate",
            "application contract",
        ],
        "M6 signposting",
        checks,
    )
    require_all(
        m9_to_m12,
        [
            "fixed-statistic required-budget calculation",
            "fix one full-split row's empirical paired means",
            "its selected $F_0$ and $i^\\star$",
            r"\Delta(w)",
            r"\mathrm{UCB}_g(n_g)=\bar d",
            r"\mathrm{bw}_g(n_g)\ge0",
            "deterministic identity",
            "does not invoke a coverage relation to an unknown mean",
            "new CAL draw",
            "not a general allocation result",
        ],
        "M9/M10 fixed-statistic scope",
        checks,
    )
    reject_all(
        m9_to_m12,
        [
            "whole feasible axis",
            "whole-axis emptiness",
            "re-select",
            "reselect",
            "resampl",
            "universal",
            "all relaxations of the per-group labels",
            "reallocating oracle true-class counts",
            r"\mathrm{UCB}_g\ge\mu_g",
            "MPB also closes the real box",
        ],
        "M9/M10 retired scope",
        checks,
    )
    require_all(
        caption,
        [
            "Fixed-statistic required-width calculation",
            "fixes the empirical paired means, $F_0$, $i^\\star$, and $\\Delta(w)$",
            "does not cover a fresh CAL sample or a changed data-derived selector",
            "relative gate (M6) is an asymptotic/descriptive diagnostic only",
        ],
        "Figure 1 caption scope",
        checks,
    )
    require_all(
        figure_text,
        [
            "fixed-statistic feasibility quadrant",
            "all 125 fixed full-split rows",
            "fixed-statistic empty",
            "stored absolute-gate content at the budget grid",
        ],
        "Figure 1 rendered-label scope",
        checks,
    )
    require_all(
        m12,
        [
            "finite stored-grid observation",
            "not a conclusion beyond those cells",
            "not an extension of the fixed-statistic M9 calculation",
        ],
        "M12 finite-grid scope",
        checks,
    )
    require_all(
        tex,
        [
            "E03 provides no application contract",
            "snapshot-reported only",
        ],
        "E03 application boundary",
        checks,
    )
    require_all(
        claims,
        [
            "corrected_revision_04_fixed_statistics_and_finite_grid_boundary",
            "Changing CAL data or the data-derived selector/margin lies outside the calculation",
        ],
        "claim map synchronization",
        checks,
    )
    require_all(
        provenance,
        [
            "M6 asymptotic relative status-quo diagnostic",
            "M9/M10 fixed-statistic required-width calculation and stored grid audit",
            "no application contract",
        ],
        "method provenance synchronization",
        checks,
    )

    passed = all(item["passed"] for item in checks)
    payload = {
        "audit": "revision_04_allocation_scope_static_audit",
        "status": "pass" if passed else "fail",
        "scope": "static source/provenance boundary check; not a scientific experiment or application-contract verification",
        "inputs": {
            "manuscript/paper.tex": sha256(TEX),
            "manuscript/fig_m10_frontier.pdf": sha256(FIGURE),
            "evidence/claim_evidence_map.tsv": sha256(CLAIMS),
            "evidence/method_provenance.tsv": sha256(PROVENANCE),
        },
        "checks": checks,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT)}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
