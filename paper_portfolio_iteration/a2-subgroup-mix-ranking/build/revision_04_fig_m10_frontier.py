#!/usr/bin/env python3
"""Render the Round-4 scoped version of the existing M10 frontier figure.

Reads the frozen M10 JSON and changes only figure labels: panel (a) is explicitly a
fixed-statistic required-width calculation, while panel (b) remains a stored-grid diagnostic.
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "remote_snapshot" / "results" / "SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json"
OUTPDF = PROJECT / "manuscript" / "fig_m10_frontier.pdf"
OUTPNG = PROJECT / "build" / "revision_04_fig_m10_frontier.png"


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    rows = [
        row
        for carrier_value in payload["cells"].values()
        for budget_value in carrier_value["budgets"].values()
        for row in budget_value["rows"]
        if not row.get("trivial", True) and float(row["budget"]) >= 0.999
    ]
    assert len(rows) == payload["emptiness_certificate"]["n_real_switch_rows_full_budget"] == 125

    carrier_col = {
        "fashion": "#d62728",
        "digits": "#1f77b4",
        "news": "#2ca02c",
        "mnist": "#9467bd",
    }
    marker_map = {"fashion": "^", "digits": "s", "news": "o", "mnist": "D"}
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.3))

    for row in rows:
        delta = row["Delta_full"]
        bandwidth_mass = delta * math.sqrt(row["bstar_hoef_full"])
        axa.scatter(
            delta,
            bandwidth_mass,
            s=14,
            color=carrier_col[row["carrier"]],
            marker=marker_map[row["carrier"]],
            alpha=0.55,
            linewidths=0,
        )
    limit = max(
        max(row["Delta_full"] for row in rows),
        max(row["Delta_full"] * math.sqrt(row["bstar_hoef_full"]) for row in rows),
    ) * 1.15
    xs = np.linspace(0, limit, 200)
    axa.plot(xs, xs, color="black", lw=1.2, ls="--", label=r"feasible boundary $\Delta=B$")
    axa.fill_between(xs, xs, limit, color="gray", alpha=0.12, label=r"fixed-statistic empty ($b^\star>1$)")
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlabel(r"fixed selector margin $\Delta(w)$")
    axa.set_ylabel(r"fixed-statistic width mass $B=\Delta\sqrt{b^\star}$")
    axa.set_title("(a) fixed-statistic feasibility quadrant\n(all 125 fixed full-split rows)", fontsize=8)
    axa.set_xlim(1e-4, limit)
    axa.set_ylim(1e-2, limit)
    axa.legend(loc="lower right", fontsize=6.5, framealpha=0.9)

    budgets = ["0.25", "0.5", "1.0"]
    carriers = ["fashion", "mnist", "digits"]
    per_carrier = payload["per_carrier_budget"]
    for carrier in carriers:
        rates = [per_carrier[carrier][budget]["abs_commit_rate"] for budget in budgets]
        axb.plot(
            [0.25, 0.5, 1.0],
            rates,
            "-",
            color=carrier_col[carrier],
            marker=marker_map[carrier],
            markersize=5,
            lw=1.4,
            label=carrier,
        )
    axb.set_xticks([0.25, 0.5, 1.0])
    axb.set_xticklabels(["0.25", "0.5", "1.0"])
    axb.set_xlabel(r"label budget $b$ (fraction of full calibration)")
    axb.set_ylabel(r"absolute-gate committed rate")
    axb.set_title("(b) stored absolute-gate content\nat the budget grid", fontsize=8)
    axb.set_ylim(0, 1.12)
    axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.legend(loc="lower right", fontsize=7, framealpha=0.9)
    axb.grid(alpha=0.25, ls=":")

    fig.tight_layout()
    fig.savefig(OUTPDF, bbox_inches="tight")
    fig.savefig(OUTPNG, dpi=160, bbox_inches="tight")

    median_delta = st.median(row["Delta_full"] for row in rows)
    median_bstar = st.median(row["bstar_hoef_full"] for row in rows)
    minimum_bstar = min(row["bstar_hoef_full"] for row in rows)
    assert abs(median_delta - 0.0139) < 0.002
    assert abs(median_bstar - 272.9) / 272.9 < 0.05
    assert minimum_bstar > 1
    print(
        json.dumps(
            {
                "source": str(SOURCE.relative_to(PROJECT)),
                "rows": len(rows),
                "median_delta": median_delta,
                "median_bstar": median_bstar,
                "minimum_bstar": minimum_bstar,
                "pdf": str(OUTPDF.relative_to(PROJECT)),
                "png": str(OUTPNG.relative_to(PROJECT)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
