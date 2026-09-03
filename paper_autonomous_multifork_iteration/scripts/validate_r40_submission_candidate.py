#!/usr/bin/env python3
"""Fail-closed static and local-evidence audit for the R40 paper candidate.

This script is intentionally read-only.  It does not compile LaTeX, run GPU
work, or treat the compact R40 dispatch mirror as a complete raw replay.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
TEX = PAPER / "main_r40_submission_candidate.tex"
BASELINE = PAPER / "main_r39_revised.tex"
REGISTRY = PAPER / "evidence/experiment_registry.json"
CLAIMS = PAPER / "evidence/claim_evidence_map.tsv"
METHODS = PAPER / "evidence/method_provenance.tsv"
SUPPLEMENT = PAPER / "supplement_r40_candidate"
DISPATCH_ID = "E-R40-PRIMARY-COMPILED-DISPATCH-V11-A"


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
    require(rows, f"empty TSV: {path}")
    require(len(rows[0]) == columns, f"unexpected TSV schema: {path}")
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


def verify_anonymous_candidate() -> int:
    ledger = SUPPLEMENT / "MANIFEST.sha256"
    rows = []
    for number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        require(match is not None, f"malformed anonymous manifest row {number}")
        expected, name = match.groups()
        path = SUPPLEMENT / name
        require(path.is_file(), f"missing anonymous candidate file: {name}")
        require(sha256_file(path) == expected, f"anonymous candidate hash drift: {name}")
        rows.append(name)
    require(len(rows) == 3 and len(set(rows)) == 3,
            "anonymous candidate manifest must close exactly three payload files")
    payload = "\n".join((SUPPLEMENT / name).read_text(encoding="utf-8") for name in rows)
    forbidden = [
        r"/Users/", r"/mnt/", r"liuhanzuo", r"trial[ _-]?\d+",
        r"job[ _-]?\d+", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    ]
    require(not any(re.search(pattern, payload, re.I) for pattern in forbidden),
            "anonymous candidate contains identifying transport metadata")
    aliases = json.loads((SUPPLEMENT / "artifact_aliases.json").read_text())
    require({row["evidence_id"] for row in aliases["aliases"]} >= {
        "E-R34-HYPIC-STORE-EXTERNAL-ACCEPTANCE", DISPATCH_ID
    }, "anonymous aliases do not cover retained-state and dispatch evidence")
    scope = json.loads((SUPPLEMENT / "dispatch_scope.json").read_text())
    require(scope["local_index_status"] == "integrity_checked_not_full_raw_replay",
            "anonymous dispatch mirror boundary drift")
    require(all(value is False for value in scope["nonclaims"].values()),
            "an anonymous dispatch nonclaim was promoted")
    return len(rows)


def main() -> None:
    require(TEX.is_file(), f"missing candidate source: {TEX}")
    require(sha256_file(BASELINE) ==
            "ef43415eb94a40b4a239c8f0fb8d3017ad9bc07ef75ff4895343292d92f546cb",
            "reviewed R39 baseline changed")
    tex = TEX.read_text(encoding="utf-8")

    forbidden_phrases = [
        "six of seven", "six complete", "dispatch remains partial",
        "dispatch-provenance coverage remains partial", "Python-call only",
        "not selected per call", "no per-call compiled-binary",
        "overall trace coverage remains partial", "trial1892234",
        "/Users/", "liuhanzuo",
    ]
    lowered = tex.lower()
    require(not any(phrase.lower() in lowered for phrase in forbidden_phrases),
            "candidate retains stale dispatch or identifying language")

    required_phrases = [
        "all seven", "209,920", "635,520", "224",
        "Triton kernel-cache artifact", "normal Python return",
        "device-side", "compiled GDN", "slot-ID-to-live-tensor binding",
        "one historical", "100.146", "850.95",
        "Reproducibility Statement", "Illustrative Deployment Store--F1",
    ]
    require(all(phrase in tex for phrase in required_phrases),
            "candidate is missing a required claim or scope boundary")

    repro = tex.index(r"\section*{Reproducibility Statement}")
    bibliography = tex.index(r"\bibliography{references}")
    appendix = tex.index(r"\appendix")
    store = tex.index(r"\label{app:illustrative-store}")
    require(repro < bibliography < appendix < store,
            "reproducibility/references/appendix order is invalid")
    abstract = tex[tex.index(r"\begin{abstract}"):tex.index(r"\end{abstract}")]
    intro = tex[tex.index(r"\section{Introduction}"):tex.index(r"\section{Background")]
    conclusion = tex[tex.index(r"\section{Conclusion}"):repro]
    require(not any("Store" in section for section in (abstract, intro, conclusion)),
            "illustrative Store--F1 result is still a headline claim")
    require(tex.index(r"\input{tables/h20_deployment_table_r40_candidate.tex}") > appendix,
            "illustrative Store--F1 table is not appendix-only")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    experiments = registry["experiments"]
    evidence_ids = [row["evidence_id"] for row in experiments]
    require(len(evidence_ids) == len(set(evidence_ids)), "duplicate evidence ID")
    dispatch = next(row for row in experiments if row["evidence_id"] == DISPATCH_ID)
    require(dispatch["active_manuscript_support"] is True, "dispatch evidence is not active")
    require(dispatch["validation"]["target_5_status_at_declared_scope"] == "pass",
            "registry does not authorize target 5 pass")
    require(dispatch["validation"]["full_remote_raw_audit_reexecuted_locally"] is False,
            "registry overstates local replay")
    require(dispatch["validation"]["standalone_postrun_auditor_identity_receipt_present"] is False,
            "registry overstates auditor identity evidence")
    for relative in dispatch["artifact_paths"]:
        require((PAPER / relative).is_file(), f"missing registered dispatch artifact: {relative}")

    claim_rows = read_tsv(CLAIMS, 7)
    claim_ids = [row["claim_id"] for row in claim_rows]
    require(len(claim_ids) == len(set(claim_ids)), "duplicate claim ID")
    require("C-R40-PRIMARY-COMPILED-DISPATCH-01" in claim_ids,
            "active R40 dispatch claim is not registered")
    require("C-UNSUPPORTED-R40-COMPILED-DISPATCH-EXPANSION" in claim_ids,
            "R40 dispatch nonclaims are not registered")
    registered_evidence = set(evidence_ids)
    for row in claim_rows:
        for evidence_id in filter(None, row["evidence_ids"].split(";")):
            require(evidence_id in registered_evidence,
                    f"claim references unknown evidence ID: {evidence_id}")

    method_rows = read_tsv(METHODS, 6)
    method_ids = [row["method_id"] for row in method_rows]
    require(len(method_ids) == len(set(method_ids)), "duplicate method ID")
    for method_id in (
        "M-R40-DISPATCH-RECEIPT", "M-R40-DISPATCH-FINALIZE",
        "M-R40-DISPATCH-INTEGRATION-AUDIT",
    ):
        require(method_id in method_ids, f"missing method provenance: {method_id}")
    for row in method_rows:
        if "{" not in row["source_path"]:
            require((PAPER / row["source_path"]).exists(),
                    f"method source path is missing: {row['method_id']}")

    asset = json.loads((PAPER / "figures/imagegen_round5_candidates/IMAGEGEN_ASSET_RECORD_R40.json").read_text())
    figure = PAPER / asset["teaser"]["output"]
    require(sha256_file(figure) == asset["teaser"]["final_raw_sha256"],
            "R40 teaser differs from its asset record")
    table = (PAPER / "tables/h20_deployment_table_r40_candidate.tex").read_text()
    require("Illustrative" in table and "not a headline" in table,
            "appendix Store--F1 table is not demoted explicitly")

    dispatch_check = subprocess.run(
        [sys.executable, str(PAPER / "scripts/validate_r40_compiled_dispatch_integration.py")],
        cwd=PAPER, text=True, capture_output=True, check=True,
    )
    dispatch_report = json.loads(dispatch_check.stdout)
    require(dispatch_report["status"] == "pass_for_local_mirror_integration",
            "local dispatch integration audit did not pass")

    report = {
        "status": "pass",
        "candidate_sha256": sha256_file(TEX),
        "included_files_checked": verify_includes(tex),
        "registry_evidence_ids": len(evidence_ids),
        "claim_rows": len(claim_rows),
        "method_rows": len(method_rows),
        "anonymous_payload_files_checked": verify_anonymous_candidate(),
        "dispatch_local_integration": dispatch_report["status"],
        "gpu_work_executed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
