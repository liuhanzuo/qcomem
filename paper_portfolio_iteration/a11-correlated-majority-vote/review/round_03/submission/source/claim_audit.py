#!/usr/bin/env python3
"""Relocatable, hash-bound audit of this reviewer-safe manuscript snapshot.

Run from any directory: ``python claim_audit.py``. It reads only this directory
and its bundled ``audit_evidence/`` folder; absent prerequisites are structured
failures. It is not a scientific rerun or proof assistant.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_root(files: dict[str, str]) -> str:
    payload = "".join(f"{files[p]}  {p}\n" for p in sorted(files))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.split())


def active_tex(text: str) -> str:
    """Remove explicitly withdrawn LaTex conditional blocks before claim checks."""
    return re.sub(r"\\iffalse.*?\\fi", "", text, flags=re.DOTALL)


def main() -> int:
    manifest = json.loads((HERE / "claim_audit_manifest.json").read_text(encoding="utf-8"))
    expected = manifest["files"]
    checks: list[tuple[str, bool, str]] = []
    actual: dict[str, str] = {}
    for rel, wanted in expected.items():
        path = HERE / rel
        if not path.is_file():
            checks.append((f"prerequisite:{rel}", False, "missing required snapshot-local file"))
            continue
        actual[rel] = digest(path)
        checks.append((f"hash:{rel}", actual[rel] == wanted, f"expected={wanted} actual={actual[rel]}"))
    checks.append(("canonical-root", len(actual) == len(expected) and canonical_root(actual) == manifest["canonical_root_sha256"], "SHA256 over sorted UTF-8 'hash  relative/path\\n' manifest lines"))

    paper_raw = (HERE / "paper.tex").read_text(encoding="utf-8")
    paper = normalized(active_tex(paper_raw))
    paper_lc = paper.lower()
    paper_title_plain = paper.replace("\\\\ ", " ")
    proofs = normalized((HERE / "appendix_proofs.tex").read_text(encoding="utf-8"))
    cmap = (HERE / "audit_evidence/claim_evidence_map.tsv").read_text(encoding="utf-8")
    visual = json.loads((HERE / "audit_evidence/visual_asset_provenance.json").read_text(encoding="utf-8"))
    fig2 = next((a for a in visual.get("assets", []) if a.get("asset_id") == "A11-FIG2-TV-CONSERVATION"), {})
    checks.extend([
        ("monotonicity-support-split", "first split off the case $\\P(F_0)=0$" in proofs and "$\\P(F_0)>0$" in proofs, "proof states zero-denominator case before ratio"),
        ("main-proof-summary", "support-boundary case" in paper and "$(32,17,29)$" in paper, "main text states repaired proof/audit scope"),
        ("tv-sensitivity-boundary", "not a population-transfer certificate" in paper and "no universal strict-refinement claim" in paper, "fitted-TV is not population containment"),
        ("tv-capacity-geometry", "capacity-constrained linear program" in paper and "piecewise linear in $R$" in paper and "$V(1)=\\max_K g(K)$" in paper and "V(R)=\\mathrm{base}_S(c)+R" not in paper, "TV LP is capacity-aware; only the full-simplex endpoint is used"),
        ("margin-exploratory-boundary", "No frozen margin-selection record is bundled" in paper, "margin thresholds are descriptive"),
        ("replay-endpoint-boundary", "pass-count replay decision" in paper and "not an aggregation of delivered textual answers" in paper, "replay endpoint remains explicit"),
        ("headline-replay-scope", "Adaptive Stopping in Count-Exchangeable Binary Pass-Count Replay" in paper_title_plain and "count-exchangeable binary pass-count replay" in paper_lc, "title and abstract name the replay-only object"),
        ("cal-theorem-assumptions", "FIT and CAL be independent i.i.d. task samples" in paper and "conditional on FIT" in paper and "TEST is read exactly once, is not used for selection" in paper, "Theorem 2 assumptions and descriptive TEST status are explicit"),
        ("table1-status", "Status. CAL-selected rows passed a FIT-frozen CAL UCB; TEST values are descriptive." in paper and "WINDOW3 has no CAL certificate" in paper, "Table 1 separates CAL selection from descriptive TEST values"),
        ("oracle-profile-capping-scope", "oracle k-wise profile-capping sensitivity calculation" in paper_lc and "final count $K$ is unobserved" in paper and "not a stopping policy" in paper_lc and "not an instance of Theorem~\\ref{thm:tower}" in paper, "K-indexed subset is explicitly an oracle diagnostic, not a theorem instance"),
        ("oracle-policy-claims-withdrawn", "universal budgets, 70-of-72 cells, domination, edge laws, crossing sets" in paper_lc and "\\label{prop:prefix}" not in active_tex(paper_raw), "policy-level cap, cost, radius, and Proposition 2 claims are excluded"),
        ("claim-map-negative-online-boundary", "C04\tThe present evidence does not establish" in cmap, "C04 remains a negative-evidence claim"),
        ("claim-map-oracle-boundary", "C09\tThe Appendix E(g) certificate-ordered" in cmap, "claim map records the oracle-only appendix boundary"),
        ("figure2-assumed-ball-boundary", "deterministic fixed-rule sensitivity radius" in paper_lc and "do not establish target-population containment" in paper_lc, "caption and limitations reject population containment"),
        ("figure2-provenance", bool(fig2) and fig2.get("sha256") == digest(HERE / "fig_tau_conservation.png") and "not a confidence set" in fig2.get("scope_boundary", ""), "Figure 2 bytes and boundary match local provenance"),
        ("figure2-renderer", "EXPECTED_INPUT_SHA256" in (HERE / "render_fig_tau_sensitivity.py").read_text(encoding="utf-8"), "renderer pins the frozen input hash"),
    ])
    failures = 0
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
        failures += not ok
    print(f"SUMMARY: {len(checks)-failures} PASS / {failures} FAIL")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
