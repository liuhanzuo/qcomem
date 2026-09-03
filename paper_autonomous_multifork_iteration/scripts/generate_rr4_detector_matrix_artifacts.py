#!/usr/bin/env python3
"""Render the preregistered RR4 detector matrix into appendix-ready LaTeX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_IDS = [f"M{index}" for index in range(1, 10)]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def detector_cell(value: Mapping[str, Any]) -> str:
    if value.get("status") != "evaluated":
        return r"\textsc{n/e}"
    return r"\checkmark" if value.get("caught") is True else r"--"


def runtime_cell(value: Mapping[str, Any]) -> str:
    require(value.get("status") == "evaluated", "runtime detector status")
    return r"\checkmark" if value.get("caught") is True else r"--"


def preservation_cell(value: str) -> str:
    mapping = {
        "output_preserved_within_measured_horizon": "preserved",
        "output_changed_within_measured_horizon": "changed",
        "not_observable_due_to_abort_or_missing_comparator": r"\textsc{n/e}",
    }
    require(value in mapping, f"unknown preservation status {value}")
    return mapping[value]


def render(summary: Mapping[str, Any]) -> str:
    require(
        summary.get("schema_version") == "forkaudit-detector-matrix-aggregate-v1",
        "aggregate schema",
    )
    rows = summary.get("rows")
    require(isinstance(rows, list), "matrix rows")
    require([row.get("mutant_id") for row in rows] == EXPECTED_IDS, "mutant coverage/order")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Detector-by-mutant ablation. A checkmark means that the detector caught the live fault; -- means it was evaluated but did not; \textsc{n/e} means not evaluated or not applicable. Output status is bounded to the preregistered measured steps. The ForkAudit column is replayed from the original RR2 raw receipt, while other columns come from the target-gate-suppressed follow-up.}",
        r"\label{tab:rr4-detector-matrix}",
        r"\begin{tabular}{llccccccl}",
        r"\toprule",
        r"ID & Fault & Token & Logit & Arm & $N$ & Runtime & ForkAudit & Output \\",
        r"\midrule",
    ]
    for row in rows:
        detectors = row["detectors"]
        original = row["original_rr2_forkaudit_receipt"]
        require(original["classification"] == "detected_expected_gate", "RR2 gate outcome")
        require(original["matched_clean_classification"] == "clean_pass", "RR2 clean")
        lines.append(
            " & ".join(
                [
                    row["mutant_id"],
                    row["mutant_name"].replace("_", r"\_"),
                    detector_cell(detectors["token_only"]),
                    detector_cell(detectors["full_logit"]),
                    detector_cell(detectors["cross_arm"]),
                    detector_cell(detectors["cross_n"]),
                    runtime_cell(detectors["existing_runtime_assertions"]),
                    r"\checkmark",
                    preservation_cell(row["output_preserving_status"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.summary.read_text(encoding="utf-8"))
    rendered = render(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(value["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
