#!/usr/bin/env python3
"""Relocatable, hash-bound audit of this reviewer-safe manuscript snapshot.

Run from any directory: ``python claim_audit.py``. It reads only this directory
and its bundled ``audit_evidence/`` folder; absent prerequisites are structured
failures. It is not a scientific rerun or proof assistant.
"""
from __future__ import annotations

import hashlib
import json
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

    paper = normalized((HERE / "paper.tex").read_text(encoding="utf-8"))
    paper_lc = paper.lower()
    proofs = normalized((HERE / "appendix_proofs.tex").read_text(encoding="utf-8"))
    cmap = (HERE / "audit_evidence/claim_evidence_map.tsv").read_text(encoding="utf-8")
    visual = json.loads((HERE / "audit_evidence/visual_asset_provenance.json").read_text(encoding="utf-8"))
    fig2 = next((a for a in visual.get("assets", []) if a.get("asset_id") == "A11-FIG2-TV-CONSERVATION"), {})
    checks.extend([
        ("monotonicity-support-split", "first split off the case $\\P(F_0)=0$" in proofs and "$\\P(F_0)>0$" in proofs, "proof states zero-denominator case before ratio"),
        ("main-proof-summary", "support-boundary case" in paper and "$(32,17,29)$" in paper, "main text states repaired proof/audit scope"),
        ("tv-sensitivity-boundary", "not a population-transfer certificate" in paper and "no universal strict-refinement claim" in paper, "fitted-TV is not population containment"),
        ("margin-exploratory-boundary", "No frozen margin-selection record is bundled" in paper, "margin thresholds are descriptive"),
        ("replay-endpoint-boundary", "pass-count replay decision" in paper and "not an aggregation of delivered textual answers" in paper, "replay endpoint remains explicit"),
        ("claim-map-negative-online-boundary", "C04\tThe present evidence does not establish" in cmap, "C04 remains a negative-evidence claim"),
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
