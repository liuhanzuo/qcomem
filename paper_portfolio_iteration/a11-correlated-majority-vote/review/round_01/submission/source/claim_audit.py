#!/usr/bin/env python3
"""Hash-bound semantic audit for the revised self-contained manuscript.

This checker deliberately does not replace ``remote_snapshot/claim_check.py``:
that frozen checker validates supplied numeric/result artifacts against the
frozen remote source.  This checker instead audits the current manuscript
package and rejects a stale manifest before checking the targeted claim
boundaries introduced in revision 01b.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.split())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "claim_audit_manifest.json",
        help="hash manifest generated for this manuscript snapshot",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(read_text(manifest_path))
    files = {
        "paper.tex": HERE / "paper.tex",
        "appendix_proofs.tex": HERE / "appendix_proofs.tex",
        "appendix_dp.tex": HERE / "appendix_dp.tex",
        "claim_audit.py": HERE / "claim_audit.py",
        "evidence/claim_evidence_map.tsv": ROOT / "evidence" / "claim_evidence_map.tsv",
        "evidence/visual_asset_provenance.json": ROOT / "evidence" / "visual_asset_provenance.json",
        "fig1_earlystop.png": HERE / "fig1_earlystop.png",
    }
    expected_files = manifest.get("files", {})
    checks: list[tuple[str, bool, str]] = []

    for label, path in files.items():
        actual = sha256(path)
        expected = expected_files.get(label)
        checks.append(
            (
                f"freshness:{label}",
                actual == expected,
                f"expected={expected} actual={actual}",
            )
        )

    frozen = ROOT / "remote_snapshot" / "paper.tex"
    frozen_actual = sha256(frozen)
    frozen_expected = manifest.get("frozen_remote_snapshot_paper_tex_sha256")
    checks.append(
        (
            "frozen-source-hash",
            frozen_actual == frozen_expected,
            f"expected={frozen_expected} actual={frozen_actual}",
        )
    )
    checks.append(
        (
            "current-source-is-not-frozen-source",
            sha256(files["paper.tex"]) != frozen_actual,
            f"current={sha256(files['paper.tex'])} frozen={frozen_actual}",
        )
    )

    paper = normalized(read_text(files["paper.tex"]))
    proofs = normalized(read_text(files["appendix_proofs.tex"]))
    dp = normalized(read_text(files["appendix_dp.tex"]))
    claim_map = list(
        csv.DictReader(
            files["evidence/claim_evidence_map.tsv"].open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    c04 = next((row for row in claim_map if row["claim_id"] == "C04"), None)
    visual_provenance = json.loads(read_text(files["evidence/visual_asset_provenance.json"]))
    fig1 = next(
        (
            asset
            for asset in visual_provenance.get("assets", [])
            if asset.get("asset_id") == "A11-FIG1-EARLYSTOP-OVERVIEW"
        ),
        None,
    )
    visual_sources = {
        item["path"]: item["sha256"]
        for item in (fig1 or {}).get("quantitative_label_audit", {}).get("sources", [])
    }
    visual_sources_ok = all(
        (ROOT / path).is_file() and sha256(ROOT / path) == expected
        for path, expected in visual_sources.items()
    )
    visual_history_sources_ok = all(
        (ROOT / item["path"]).is_file()
        and sha256(ROOT / item["path"]) == item["sha256"]
        for item in (fig1 or {}).get("source_assets", [])
    )
    fig1_history = json.loads(read_text(ROOT / "remote_snapshot" / "FIG1_HISTORY.json"))
    edit_script = ast.parse(
        read_text(ROOT / "remote_snapshot" / "edit_a11_earlystop_fig1.py")
    )
    edit_constants = {
        target.id: ast.literal_eval(node.value)
        for node in edit_script.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"GEN_PROMPT", "EDIT_PROMPT_HEADER"}
    }
    provenance_history = (fig1 or {}).get("generation_and_edit_history", {})
    provenance_steps = provenance_history.get("steps", [])
    visual_prompt_history_ok = (
        fig1 is not None
        and fig1.get("prompt") == edit_constants.get("GEN_PROMPT")
        and provenance_history.get("edit_prompt_header", {}).get("verbatim_text")
        == edit_constants.get("EDIT_PROMPT_HEADER")
        and [
            {key: step.get(key) for key in ("step", "src", "dst", "prompt")}
            for step in provenance_steps
        ]
        == fig1_history.get("steps")
        and all(
            step.get("fixes_verbatim")
            == read_text(ROOT / step["fixes_file"])
            for step in provenance_steps
            if step.get("fixes_file")
        )
    )
    fit_result = json.loads(
        read_text(ROOT / "remote_snapshot" / "results" / "fit_cal_test_r469_result.json")
    )
    margin_result = json.loads(
        read_text(ROOT / "remote_snapshot" / "results" / "margin_repair_r469_result.json")
    )
    passrate_result = json.loads(
        read_text(ROOT / "remote_snapshot" / "results" / "passrate_r467_result.json")
    )
    visual_values_ok = (
        fit_result["test_readout"]["FULL32"]["saving_vs_full"] == 0.0
        and fit_result["test_readout"]["FIXED_HOEF_a0.05"]["saving_vs_full"] == 0.1562
        and fit_result["test_readout"]["FIXED_EB_a0.05"]["saving_vs_full"] == 0.4688
        and fit_result["test_readout"]["BAYESH_a0.05"]["saving_vs_full"] == 0.8085
        and fit_result["test_readout"]["BAYESH_a0.05"]["realized_flip"] == 0.02489
        and fit_result["fair_gap_bayesh_vs_hoeffding"]["0.05"]["abs_gap_bayesh_minus_hoef"] == 0.6522
        and fit_result["fair_gap_bayesh_vs_hoeffding"]["0.05"]["significant"] is True
        and margin_result["results"]["a0.05_E3_blockswap_d0.15_g0.025"]["saving"] == 0.7716
        and margin_result["results"]["a0.05_E3_blockswap_d0.15_g0.025"]["valid"] is True
        and margin_result["results"]["a0.05_E3_blockswap_d0.15_g0.0"]["saving"] == 0.8102
        and passrate_result["frac_extreme_p_le0.1_or_ge0.9"] == 0.4492
    )

    checks.extend(
        [
            (
                "main-terminal-certificate",
                r"\cert(x_N,N)=0" in paper,
                "paper.tex must state the reachable terminal certificate explicitly",
            ),
            (
                "main-binary-pass-count-boundary",
                "binary pass-count replay decision" in paper
                and "not an aggregation of delivered textual answers" in paper
                and "actual delivered answer-majority" in paper,
                "paper.tex must distinguish the replay decision from delivered-answer aggregation",
            ),
            (
                "theorem-terminal-threshold",
                "terminal state $k=N$ satisfies this same rule" in paper
                and "At $\\tau=N$, $x_N=K$" in paper,
                "Theorem 1/proof must treat k=N as a zero-certificate threshold state",
            ),
            (
                "proof-terminal-zero",
                r"\cert(x_N,N)=0" in proofs
                and "The first term is at most" in proofs
                and "the second term is zero" in proofs,
                "appendix proof must derive a zero terminal contribution",
            ),
            (
                "dp-terminal-zero",
                r"\cert(x,N):=0" in dp
                and "same stopping test" in dp
                and "zero for replay flip" in dp,
                "DP appendix must use the same terminal semantics",
            ),
            (
                "obsolete-forced-stop-language-absent",
                all(
                    phrase not in " ".join((paper, proofs, dp))
                    for phrase in (
                        "certificate may exceed $\\alpha$",
                        "absorbed into the population bound",
                        "Conditional per-problem validity at forced stops is not guaranteed",
                    )
                ),
                "obsolete exceptional-forced-stop explanation must be absent",
            ),
            (
                "c04-negative-evidence-wording",
                c04 is not None
                and c04["claim"].startswith("The present evidence does not establish")
                and c04["status"] == "unsupported / must not claim",
                "C04 must be a negative evidence statement with the unsupported status",
            ),
            (
                "figure-caption-mixed-role-disclosure",
                "Conceptual overview with artifact-checked quantitative labels." in paper
                and "numbers are rounded reproductions of the named replay artifacts" in paper,
                "Figure 1 caption must distinguish conceptual composition from its evidence-linked labels",
            ),
            (
                "ai-use-figure-disclosure",
                "quantitative labels were independently checked against" in paper
                and "not generate, execute, or replace" in paper,
                "AI-use statement must disclose the generated layout and independent label check",
            ),
            (
                "visual-provenance-evidence-dependency",
                fig1 is not None
                and fig1.get("evidence_dependency") is True
                and fig1.get("caption_marks_illustrative") is True
                and fig1.get("sha256") == sha256(files["fig1_earlystop.png"])
                and fig1.get("quantitative_label_audit", {}).get("status") == "pass",
                "visual provenance must bind the current PNG and disclose quantitative dependence",
            ),
            (
                "visual-quantitative-artifact-crosscheck",
                visual_sources_ok and visual_values_ok,
                "frozen result hashes and all Figure 1 rounded quantitative labels must match their registered JSON values",
            ),
            (
                "visual-prompt-edit-history-crosscheck",
                visual_history_sources_ok and visual_prompt_history_ok,
                "frozen Figure 1 source hashes, verbatim generation/edit prompts, and recorded chronology must match",
            ),
        ]
    )

    failed = 0
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}")
        failed += not passed
    print(f"SUMMARY: {len(checks) - failed} PASS / {failed} FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
