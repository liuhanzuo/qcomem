#!/usr/bin/env python3
"""Render a non-comparable published-systems context table from verified sources."""

from __future__ import annotations

import json
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
SOURCE = PAPER / "literature/reported_system_context.json"
TABLE = PAPER / "tables/related_work_reported_context.tex"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    require(
        payload.get("schema_version") == "forkaudit-reported-systems-context-v1",
        "reported-context schema drift",
    )
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == 9, "reported-context row drift")
    keys = [row.get("citation_key") for row in rows]
    require(len(set(keys)) == len(keys), "citation keys must be unique")
    for row in rows:
        require(
            set(row)
            == {
                "system",
                "citation_key",
                "scope",
                "native_setting",
                "reported_result",
                "quality_context",
                "primary_url",
            },
            "reported-context row schema drift",
        )
        require(str(row["primary_url"]).startswith("https://"), "primary URL drift")

    lines = [
        r"\begin{table}[H]",
        r"\caption{Unpooled published-systems context in each paper's native protocol.  These values show the scale and evaluation breadth of related systems, \emph{not} a cross-paper leaderboard or ForkAudit evidence: models, hardware, batching, baselines, and metrics differ.  Consequently, no row is merged with our same-stack H20 Table~\ref{tab:h20-deployment}, and the absence of a common score must not be read as a negative result.}",
        r"\label{tab:published-context}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.15\textwidth}>{\raggedright\arraybackslash}p{0.28\textwidth}>{\raggedright\arraybackslash}p{0.265\textwidth}>{\raggedright\arraybackslash}p{0.245\textwidth}@{}}",
        r"\toprule",
        r"System & Native method and study setting & Reported efficiency in its native protocol & Quality and comparability boundary \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['system']}~\\citep{{{row['citation_key']}}} & "
            f"{row['scope']}\\newline \\textit{{Setting:}} {row['native_setting']} & "
            f"{row['reported_result']} & {row['quality_context']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    TABLE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(TABLE)}, sort_keys=True))


if __name__ == "__main__":
    main()
