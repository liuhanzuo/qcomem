#!/usr/bin/env python3
"""r506: unified discrete-stopping-set geometry note — one schema, one
analytical reconciliation, one machine-checked consistency claim.

PREREGISTRATION (written before any output was inspected):
- Sources (all frozen artifacts, zero new rollouts, zero TEST/CAL reads;
  every quantity below is a FIT-grid quantity):
  (a) ../earlystop_drift_r501/fit_pareto_r501_result.json
      -> claim A: tau(cap) violated monotonicity on 323 adjacent-cap
         comparisons across the coarse 12-cap grid (preregistered C4 refuted);
         the r501 JSON text prints the denominator as 1728 but the
         comparisons number 72x11=792 (housekeeping erratum, see U3);
  (b) ../earlystop_drift_r503/critical_cap_r503_result.json
      -> claim B: on the fine grid the full-radius region is a prefix
         [0, c*] with c* >= alpha in all 72 cells (P1/P2 pass);
  (c) ../earlystop_drift_r498/efficiency_r498c_result.json
      -> claim C: deadline repair RLVE m=187 a=.05: tau 0.235 -> 0.133;
  (d) ../earlystop_drift_r505/edge_tightness_r505_result.json
      -> claim D: 48 OMR cells split into 11 tight / 37 strict by the
         sign of the slack one step below the closed-form edge;
  (e) ../earlystop_drift_r498/diag_kappa_rlve_r498d.log (text log, regex
      parsed) -> probed atom flip-by-deadline sequence
      kappa=8..3: 0.0143/0.3571/0.2143/0.5/0.2429/0.5.
- Question: are A and B consistent? Analytical answer to be verified:
  tau(cap) is NOT a monotone function in general, but its crossing set
  {cap : tau(cap)=1} is DOWNWARD-CLOSED (a prefix), because cap c' <= c
  dominates cap c in BOTH scalarizations that tau reads:
  gmax_S(c') <= gmax_S(c) (the excluded prefix is certificate-ascending)
  and base_S(c') <= base_S(c) (r501 C3, 0 violations on 1728 pairs).
  Hence once tau(c)=1 both scalar bounds are below alpha and stay below
  for every smaller cap; interior non-monotonicity lives strictly inside
  the sub-unit region and cannot produce a hole in the full-radius set.
- Checks (self-falsifying):
  U1: fine-grid prefix property — for every one of the 72 cells, the set
      of scanned caps with tau >= 1-1e-6 is exactly {cap <= c*}, i.e. no
      scanned cap above c* has tau=1 and no scanned cap at or below c*
      has tau<1 (grid-level certificate of downward-closedness;
      0 holes / 0 overhangs).
  U2: fine-grid joint scalarization monotonicity — along the fine grid,
      gmax_S nondecreasing AND base_S-consistent (gmax_S nondecreasing
      already implies the crossing scalar; we verify gmax_S 0 violations
      on all adjacent fine-grid pairs, 72 cells x ~38 pairs).
  U3: coarse-grid interior non-monotonicity reproduced — recomputing the
      C4 comparison count from the r501 artifact frontier rows / raw
      per-cap table equals 323 (byte-anchored to the printed number).
  U4: kappa probe sequence anchored — parsed flips equal the printed
      [0.0143, 0.3571, 0.2143, 0.5, 0.2429, 0.5] (4dp), and the sequence
      is non-monotone (>=1 strict up-jump as kappa decreases).
  U5: r498c deadline tau_real 0.133 vs tau_orig 0.235 (f3) anchor.
- Falsification policy: any check failure is reported verbatim in this
  JSON (checks dict) and the note text degrades to the surviving subset;
  no number is edited to fit.
Output: discrete_geometry_r506_result.json.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)

R501 = os.path.join(WS, "earlystop_drift_r501", "fit_pareto_r501_result.json")
R503 = os.path.join(WS, "earlystop_drift_r503", "critical_cap_r503_result.json")
R498C = os.path.join(WS, "earlystop_drift_r498", "efficiency_r498c_result.json")
R505 = os.path.join(HERE, "..", "earlystop_drift_r505",
                    "edge_tightness_r505_result.json")
R498E = os.path.join(WS, "earlystop_drift_r498", "full_sweep_r498e_result.json")
KAPPA_LOG = os.path.join(WS, "earlystop_drift_r498",
                         "diag_kappa_rlve_r498d.log")

out = {"source": "frozen r501/r503/r498c/r505/r498e artifacts; zero new data; "
                 "zero TEST/CAL reads; deterministic",
       "prereg": ("U1 fine-grid prefix (0 holes/0 overhangs); U2 gmax_S "
                  "nondecreasing on fine grid; U3 C4 count == 323; U4 kappa "
                  "probe sequence anchored and non-monotone; U5 deadline "
                  "0.235->0.133 anchor"),
       "checks": {}}

# ---------- U1/U2 on r503 fine grid ----------
d503 = json.load(open(R503))
holes = 0
overhangs = 0
gmax_viol = 0
n_pairs = 0
n_cells = 0
for cname, cblk in d503["carriers"].items():
    for cell in cblk["cells"]:
        n_cells += 1
        rows = sorted(cell["curve"], key=lambda r: r["cap"])
        cstar = cell["c_star"]
        for r in rows:
            is_full = r["tau"] >= 1.0 - 1e-6
            if r["cap"] <= cstar + 1e-12 and not is_full:
                holes += 1
            if r["cap"] > cstar + 1e-9 and is_full:
                overhangs += 1
        prev = None
        for r in rows:
            if prev is not None:
                n_pairs += 1
                if r["gmax_S"] < prev["gmax_S"] - 1e-9:
                    gmax_viol += 1
            prev = r
out["checks"]["U1_fine_prefix"] = {
    "cells": n_cells, "holes": holes, "overhangs": overhangs,
    "pass": n_cells == 72 and holes == 0 and overhangs == 0}
out["checks"]["U2_gmax_nondecr_fine"] = {
    "adjacent_pairs": n_pairs, "violations": gmax_viol,
    "pass": gmax_viol == 0}

# ---------- U3: C4 count reproduced from r498e raw table (independent
# recomputation of the r501 printed 323/1728) ----------
d498e = json.load(open(R498E))
caps_desc = list(d498e["caps"])  # swept descending (as r501 iterated)
cells498 = []
for cname, cblk in d498e["carriers"].items():
    for cell in cblk["cells"]:
        cells498.append((cname, cell))
c4 = 0
c3 = 0
total = 0
for cname, c in cells498:
    prev_tau = None
    prev_base = None
    for cap in caps_desc:
        e = c["per_cap"][str(cap)]
        if prev_tau is not None:
            total += 1
            if e["tau"] > prev_tau + 1e-9:
                c4 += 1
            if e["base_S"] > prev_base + 1e-12:
                c3 += 1
        prev_tau = e["tau"]
        prev_base = e["base_S"]
d501 = json.load(open(R501))
# NOTE (r506 erratum on r501/r502 housekeeping): the printed denominator
# "1728" in the r501 artifact text and the R501/R502/R505 log/status text is
# wrong; the adjacent-cap comparisons number 72 cells x 11 adjacent pairs =
# 792. The numerator 323 reproduces exactly. Nothing downstream reads the
# denominator (the paper never printed it; the r501 checks gate only on
# C2/C3/C5), so this is a housekeeping erratum, disclosed here and in the
# r506 round memo; the mutable-line note anchors the 792 figure.
out["checks"]["U3_c4_count"] = {
    "recomputed_c4": c4, "printed_c4": 323, "pairs": total,
    "r501_printed_pairs_erratum": "printed 1728 in r501 JSON/memo; true 792",
    "r501_checks_all_pass": d501["checks"]["ALL_PASS"] is True,
    "c3_violations_recomputed": c3,
    "pass": c4 == 323 and total == 792 and c3 == 0
            and d501["checks"]["ALL_PASS"] is True}

# ---------- U4: kappa probe sequence from the r498d text log ----------
flips = None
with open(KAPPA_LOG) as f:
    lines = f.read().splitlines()
for i, ln in enumerate(lines):
    if ln.startswith("K=4 "):
        seq = []
        for j in range(i + 1, i + 7):
            m = re.search(r"kappa=(\d+): flip=([0-9.]+)", lines[j])
            seq.append((int(m.group(1)), float(m.group(2))))
        flips = [v for _, v in sorted(seq, key=lambda t: -t[0])]
        break
PRINTED = [0.0143, 0.3571, 0.2143, 0.5, 0.2429, 0.5]
upjumps = sum(1 for a, b in zip(flips, flips[1:]) if b > a + 1e-9)
out["checks"]["U4_kappa_probe"] = {
    "parsed_4dp": [round(v, 4) for v in flips],
    "printed_4dp": PRINTED,
    "strict_upjumps_as_kappa_decreases": upjumps,
    "pass": [round(v, 4) for v in flips] == PRINTED and upjumps >= 2}

# ---------- U5: r498c deadline anchor ----------
d498c = json.load(open(R498C))
cell = next(c for c in d498c["cells"]
            if c["carrier"] == "rlve" and c["m"] == 187
            and abs(c["alpha"] - 0.05) < 1e-12)
out["checks"]["U5_deadline_anchor"] = {
    "tau_orig_f3": round(cell["tau_orig"], 3),
    "tau_real_f3": round(cell["tau_real"], 3),
    "pass": (round(cell["tau_orig"], 3) == 0.235
             and round(cell["tau_real"], 3) == 0.133
             and cell["tau_real"] < cell["tau_orig"])}

# ---------- D: tightness regime anchor (r505, read-only cross-reference)
d505 = json.load(open(os.path.abspath(R505)))
out["checks"]["U6_tightness_regimes"] = {
    "n_tight_1e-6": d505["checks"]["P1_corrected_count"]["n_tight_1e-6"],
    "n_cells": len(d505["cells"]),
    "all_r505_checks_pass": d505["checks"]["ALL_PASS"] is True,
    "pass": (d505["checks"]["P1_corrected_count"]["n_tight_1e-6"] == 11
             and len(d505["cells"]) == 48
             and d505["checks"]["ALL_PASS"] is True)}

out["checks"]["ALL_PASS"] = all(v.get("pass") for v in out["checks"].values()
                                if isinstance(v, dict) and "pass" in v)

with open(os.path.join(HERE, "discrete_geometry_r506_result.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out["checks"], indent=1))
print("ALL_PASS", out["checks"]["ALL_PASS"])
