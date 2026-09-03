#!/usr/bin/env python3
"""Hash-bound Round-4 scope audit for the reviewer-safe manuscript snapshot.

This is a static provenance and claim-boundary audit, not a scientific rerun or
proof assistant. Run from any directory with python3 claim_audit.py.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_root(files: dict[str, str]) -> str:
    payload = "".join(f"{files[p]}  {p}\n" for p in sorted(files))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    manifest = json.loads((HERE / "claim_audit_manifest.json").read_text(encoding="utf-8"))
    expected: dict[str, str] = manifest["files"]
    checks: list[tuple[str, bool, str]] = []
    actual: dict[str, str] = {}
    for rel, wanted in expected.items():
        path = HERE / rel
        if not path.is_file():
            checks.append((f"prerequisite:{rel}", False, "missing"))
            continue
        actual[rel] = digest(path)
        checks.append((f"hash:{rel}", actual[rel] == wanted, f"expected={wanted} actual={actual[rel]}"))
    checks.append((
        "canonical-root",
        len(actual) == len(expected) and canonical_root(actual) == manifest["canonical_root_sha256"],
        "SHA-256 over sorted UTF-8 hash/relative-path manifest lines",
    ))

    raw = (HERE / "paper.tex").read_text(encoding="utf-8")
    # Claims routinely wrap across TeX source lines.  Normalize whitespace for
    # semantic checks, while retaining raw text for cite parsing and duplicate
    # baseline checks below.
    paper = " ".join(raw.split())
    cmap = (HERE / "audit_evidence/claim_evidence_map.tsv").read_text(encoding="utf-8")
    visual = json.loads((HERE / "audit_evidence/visual_asset_provenance.json").read_text(encoding="utf-8"))
    assets = {a.get("asset_id"): a for a in visual.get("assets", [])}
    fig1 = assets.get("A11-FIG1-R4-ORACLE-FIT-CAL-SCHEMATIC", {})
    fig2 = assets.get("A11-FIG2-TV-CONSERVATION", {})
    fig1_inputs = json.loads(
        (HERE / "audit_evidence/results/fit_cal_test_r469_result.json").read_text(encoding="utf-8")
    )
    citation_lock = json.loads((ROOT / "literature/citation_lock.json").read_text(encoding="utf-8"))
    citation_requests = json.loads((ROOT / "literature/citation_requests.json").read_text(encoding="utf-8"))
    active_cites = {
        key.strip()
        for group in re.findall(r"\\cite[a-z]*\{([^}]+)\}", raw)
        for key in group.split(",")
    }
    locked_cites = {entry["citation_key"] for entry in citation_lock["entries"] if entry["status"].startswith("locked")}
    request_status = {entry["citation_key"]: entry["status"] for entry in citation_requests["requests"]}

    forbidden = (
        "most robust",
        "between-carrier law",
        "unified mechanism",
        "certificate-ordered subset",
        "dominant deviation from i.i.d. is not latent",
    )
    checks.extend([
        (
            "oracle-identity-true-H-only",
            "Oracle conditional certificate (true $\\trueH$ only)" in paper
            and "Under the true law $\\trueH$" in paper
            and "Oracle conditional identity and adaptive bound" in paper,
            "Theorem 1 is explicitly an oracle true-H statement",
        ),
        (
            "fitted-score-distinct",
            "Fitted implementation (BAYES-H)" in paper
            and "fitted stopping score" in paper
            and "unless $\\widehat H_{\\mathrm{FIT}}=\\trueH$" in paper,
            "BAYES-H is named a distinct plug-in fitted score",
        ),
        (
            "theorem2-only-operational",
            "CAL-screened marginal replay guarantee for a FIT-frozen rule family" in paper
            and "The operational guarantee for this fitted rule is" in paper
            and "not Theorem~\\ref{thm:tower}" in paper,
            "Fitted implementation points only to Theorem 2",
        ),
        (
            "cal-sampling-conditions",
            "Let FIT and CAL be independent i.i.d. task samples" in paper
            and "conditional on FIT" in paper
            and "bounded i.i.d. per-task flips" in paper,
            "Theorem 2 names independent i.i.d. FIT/CAL and bounded task loss",
        ),
        (
            "bonferroni-J64",
            "60 candidates" in paper
            and "per-rule $\\delta_r=0.05/64$" in paper
            and "$J=64$ FIT-frozen candidates" in paper,
            "Protocol fixes the active finite family and Bonferroni count",
        ),
        (
            "test-descriptive-only",
            "TEST is read exactly once" in paper
            and "descriptive readout rather than a second population guarantee" in paper,
            "TEST scope is explicit",
        ),
        (
            "e3-counterexample",
            "BAYES-H is invalid ($0.052$) while FIXED-EB remains valid" in paper
            and "Every margin-exploration result is" in paper,
            "E3 alpha=.05 delta=.15 is retained as an explicit synthetic counterexample",
        ),
        (
            "synthetic-margin-boundary",
            "synthetic, exploratory, and nonconfirmatory" in paper
            and "No frozen margin-selection record is bundled" in paper,
            "Margin exploration is not promoted to confirmation",
        ),
        (
            "openr1-accounting",
            "9{,}374 raw unique problems" in paper
            and "exactly two deduplicated rollouts" in paper
            and "$3000/3000/2853$" in paper,
            "Raw/analyzed OpenR1 populations and split are distinguished",
        ),
        (
            "derived-count-boundary",
            "int(round(float(pass\\_rate\\_72b\\_tir) * 32))" in paper
            and "does not expose the raw parquet, raw evaluator, or answer-extraction path" in paper,
            "Carrier text limits count provenance to the frozen derivation",
        ),
        (
            "no-correlation-dominance-claim",
            "cannot identify whether within-task correlation is the dominant deviation" in paper,
            "Count-only artifact does not identify correlation dominance",
        ),
        (
            "table1-no-global-best-bold",
            "BAYES-H & 0.10 & \\textbf{0.0370} & 5.1 & 84.0\\%" in paper,
            "Alpha=.10 BAYES-H saving is not bolded above FIXED-EB's 84.4%",
        ),
        (
            "baselines-no-duplicate-bayes-unif",
            raw.count("BAYES-UNIF (uniform prior;") == 1,
            "BAYES-UNIF appears exactly once in the active baselines",
        ),
        (
            "table5-cal-eb-ucb",
            "Carrier & rule & $\\alpha$ & CAL EB UCB" in paper
            and "FIT-frozen and Bonferroni-adjusted" in paper,
            "Evidence matrix uses CAL EB UCB rather than an unlabeled certificate",
        ),
        (
            "no-prohibited-superlatives-or-laws",
            not any(term in paper.lower() for term in forbidden),
            "No active source uses withdrawn robustness/mechanism language",
        ),
        (
            "closest-work-citations",
            "WaudbySmithRamdas2020Confidence" in paper
            and "RossellMuller2013SequentialStopping" in paper
            and "Novikov2010OptimalSequential" in paper,
            "Closest finite-population and Bayesian-stopping sources are cited",
        ),
        (
            "closest-work-novelty-scope",
            "Our supported novelty is narrower" in paper
            and "exact task-level replay loss" in paper,
            "Novelty is restricted to frozen plug-in FIT/CAL screening",
        ),
        (
            "citation-lock-active-keys",
            active_cites <= locked_cites,
            "Every active citekey has a locked support-scoped citation entry",
        ),
        (
            "closest-work-request-lock-consistency",
            all(request_status.get(key) == "verified" for key in (
                "RossellMuller2013SequentialStopping",
                "Novikov2010OptimalSequential",
                "WaudbySmithRamdas2020Confidence",
            )),
            "Round-4 closest-work requests agree with the citation lock",
        ),
        (
            "repro-bundle-boundary",
            "byte-for-byte (SHA-256" in paper
            and "not reconstruct the manifest from the omitted raw parquet" in paper
            and "rather than locally rerun" in paper,
            "Paper states conditional replay and omitted raw/secondary boundary",
        ),
        (
            "claim-map-core-boundaries",
            all(tag in cmap for tag in ("C01", "C03", "C06", "C08", "C10", "C11", "C12", "C13")),
            "Evidence map includes oracle, CAL, recovery, drift, OpenR1, literature, visual, and negative-evidence entries",
        ),
        (
            "fig1-active-provenance",
            bool(fig1)
            and fig1.get("used_in_manuscript") is True
            and fig1.get("renderer", {}).get("mode") == "code-native Matplotlib; no generative image model"
            and fig1.get("sha256", {}).get("pdf") == digest(HERE / "fig1_round4.pdf")
            and fig1.get("sha256", {}).get("png") == digest(HERE / "fig1_round4.png"),
            "Active Figure 1 bytes and code-native provenance match",
        ),
        (
            "fig1-input-manifest-and-renderer",
            fig1.get("renderer", {}).get("sha256") == digest(HERE / "render_fig1_round4.py")
            and fig1.get("input_manifest", {}).get("sha256") == digest(HERE / "fig1_round4_input_manifest.json")
            and "EXPECTED_INPUT_SHA256" in (HERE / "render_fig1_round4.py").read_text(encoding="utf-8"),
            "Renderer and input manifest are hash-pinned",
        ),
        (
            "fig1-half-up-percent-rounding",
            "ROUND_HALF_UP" in (HERE / "render_fig1_round4.py").read_text(encoding="utf-8")
            and fig1_inputs["test_readout"]["BAYESH_a0.05"]["saving_vs_full"] == 0.8085
            and "0.8085 -> 80.9%" in (HERE / "fig1_round4_input_manifest.json").read_text(encoding="utf-8"),
            "Figure labels use the manuscript half-up convention for 0.8085",
        ),
        (
            "fig1-required-scope-labels",
            "random replay prefix" in paper
            and "chronology" in paper
            and "synthetic, exploratory, and nonconfirmatory" in paper,
            "Caption/source retain required replay and synthetic boundaries",
        ),
        (
            "fig1-caption-layout-match",
            "(b, top) only the true count law" in paper
            and "(b, middle) BAYES-H uses the distinct fitted score" in paper
            and "(b, left)" not in paper
            and "(b, right)" not in paper,
            "Caption directions match the compact stacked oracle/fitted panel",
        ),
        (
            "historical-figure-retained-not-active",
            assets.get("A11-FIG1-EARLYSTOP-HISTORICAL", {}).get("used_in_manuscript") is False,
            "Legacy generated Figure 1 is explicitly historical-only",
        ),
        (
            "fig1-effective-type-at-page-width",
            "8.25pt" in fig1.get("accessibility_check", "")
            and "approximately 6.9pt effective" in fig1.get("accessibility_check", ""),
            "Active Figure 1 records its compact two-row effective-type calculation",
        ),
        (
            "fig2-sensitivity-boundary",
            bool(fig2)
            and fig2.get("sha256") == digest(HERE / "fig_tau_conservation.png")
            and "not a confidence set" in fig2.get("scope_boundary", ""),
            "Figure 2 remains an assumed-ball sensitivity diagnostic",
        ),
    ])

    failures = 0
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
        failures += not ok
    print(f"SUMMARY: {len(checks) - failures} PASS / {failures} FAIL")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
