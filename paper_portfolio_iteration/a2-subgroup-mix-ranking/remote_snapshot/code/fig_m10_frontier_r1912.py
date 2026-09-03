#!/usr/bin/env python3
"""Two-panel certificate-time figure for the M10 exact-band emptiness result (r1912).
Reads ONLY the frozen M10 JSON results/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json.
No new data is created; the figure is a pure visualization of existing certificate rows.

Panel (a): selector-margin �" per-group bandwidth-mass quadrant for the 125 real-switch
    rows at the full split (b=1.0). x=Delta_full (selector margin), y=B=Delta*sqrt(bstar)
    (a scalar proportional to the tightest Hoeffding bandwidth-mass required for admission).
    Feasible admission requires Delta >= B  (i.e. bstar<=1), so everything above the
    y=x line is the provably-empty region; all 125 rows lie there.
Panel (b): absolute exact-gate (M2.5) committed rate vs label budget b, per carrier,
    with the no-worse-than-F0 coverage on committed rows. Shows the deployable rule keeps
    content monotonically as budget grows (the relative gate does not, Sec app:tau M10).
"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JSONPATH = os.path.join(ROOT, "results", "SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json")
OUTPDF = os.path.join(ROOT, "paper", "fig_m10_frontier.pdf")
OUTPNG = os.path.join(ROOT, "paper", "fig_m10_frontier.png")

d = json.load(open(JSONPATH))
cells = d["cells"]

# ---- Panel (a): full-budget real-switch rows ----
rows = []
for cv in cells.values():
    for bkv in cv["budgets"].values():
        if float(bkv["n_per_group"].get("0") or 0) == 0:
            pass
        for r in bkv["rows"]:
            if not r.get("trivial", True) and float(r["budget"]) >= 0.999:
                rows.append(r)
assert len(rows) == d["emptiness_certificate"]["n_real_switch_rows_full_budget"] == 125, \
    f"row count mismatch: {len(rows)}"

carrier_col = {"fashion": "#d62728", "digits": "#1f77b4", "news": "#2ca02c", "mnist": "#9467bd"}
marker_map = {"fashion": "^", "digits": "s", "news": "o", "mnist": "D"}

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.3))

for r in rows:
    D = r["Delta_full"]
    B = D * math.sqrt(r["bstar_hoef_full"])
    axa.scatter(D, B, s=14, color=carrier_col[r["carrier"]],
                marker=marker_map[r["carrier"]], alpha=.55, linewidths=0)
# feasible/empty boundary
lim = max(max(r["Delta_full"] for r in rows), max(r["Delta_full"]*math.sqrt(r["bstar_hoef_full"]) for r in rows))*1.15
xs = np.linspace(0, lim, 200)
axa.plot(xs, xs, color="black", lw=1.2, ls="--", label=r"feasible boundary $\Delta=B$")
axa.fill_between(xs, xs, lim, color="gray", alpha=.12,
                 label="provably empty ($b^\star>1$)")
axa.set_xscale("log"); axa.set_yscale("log")
axa.set_xlabel(r"selector margin $\Delta(w)$")
axa.set_ylabel(r"required bandwidth-mass  $B=\Delta\sqrt{b^\star}$")
axa.set_title("(a) exact-band feasibility quadrant\n(all 125 real-switch rows)", fontsize=8)
axa.set_xlim(1e-4, lim); axa.set_ylim(1e-2, lim)
axa.legend(loc="lower right", fontsize=6.5, framealpha=.9)

# ---- Panel (b): absolute exact gate, committed rate vs budget ----
budgets = ["0.25", "0.5", "1.0"]
carriers = ["fashion", "mnist", "digits"]
pb = d["per_carrier_budget"]
for c in carriers:
    yc = [pb[c][b]["abs_commit_rate"] for b in budgets]
    axb.plot([0.25,0.5,1.0], yc, "-", color=carrier_col[c], marker=marker_map[c],
             markersize=5, lw=1.4, label=c)
axb.set_xticks([0.25,0.5,1.0]); axb.set_xticklabels(["0.25","0.5","1.0"])
axb.set_xlabel(r"label budget $b$ (fraction of full calibration)")
axb.set_ylabel(r"absolute-gate committed rate")
axb.set_title("(b) absolute exact gate keeps content\nmonotonically in budget", fontsize=8)
axb.set_ylim(0, 1.12)
axb.set_yticks([0,.25,.5,.75,1.0])
axb.legend(loc="lower right", fontsize=7, framealpha=.9)
axb.grid(alpha=.25, ls=":")

fig.tight_layout()
os.makedirs(os.path.dirname(OUTPDF), exist_ok=True)
fig.savefig(OUTPDF, bbox_inches="tight")
fig.savefig(OUTPNG, dpi=150, bbox_inches="tight")
print("wrote", OUTPDF)
print("wrote", OUTPNG)

# ---- inline honesty check (assert vs JSON, not fabrication) ----
import statistics as st
Dm = st.median(r["Delta_full"] for r in rows)
bm = st.median(r["bstar_hoef_full"] for r in rows)
bmin = min(r["bstar_hoef_full"] for r in rows)
print(f"rows={len(rows)}  Delta_med={Dm:.4f}  b_med={bm:.1f}  b_min={bmin:.2f}")
assert abs(Dm-0.0139) < 0.002, "Delta median drift"
assert abs(bm-272.9)/272.9 < 0.05, "bstar median drift"
assert all(r["bstar_hoef_full"] > 1 for r in rows), "not all rows infeasible"
print("OK all rows infeasible; medians match frozen paper prose")