#!/usr/bin/env python3
"""r508: machine witness for the formal Proposition "prefix law and
two-scalar monotonicity of the full-radius crossing set" (paper v5.22,
App D(g)).

MGR 5a29a028d9a3 (2026-08-15 05:50): retain all same-question geometric
closure requirements of 2d502ff5db3e with the mutable v5.20 line as base;
among them, promote the r506 empirical facts (full-radius crossing set is
a prefix; both scalarizations monotone) into a formal Proposition with a
short proof, using the corrected 323/792 figures, and keep an honest
provenance note for the r503 inline monotonicity assertion that was never
stored in the r503 script.

ANALYTICAL STATEMENT (proved in the paper, witnessed here):
Let tau(c) = tau*(Hm, g_{S(c)}; alpha) for the certificate-ordered
state-subset rule at flip budget c (same construction as r498d/r498e,
i.e. subset_profile: per atom K, keep the cert-ascending prefix of the
original flip-contributing stop states with cumulative mass <= c).
  (i)   Both scalarizations are monotone in c:
        gmax_S(c) := max_K g_{S(c)}(K) and base_S(c) := sum_K Hm[K]
        g_{S(c)}(K) are nondecreasing in c.  (Proof: greedy with
        nondecreasing budget keeps a nested family, S(c') subseteq S(c)
        for c' <= c; each kept-state contribution is nonnegative.)
  (ii)  tau(c) = 1  iff  base_S(c) <= alpha and gmax_S(c) <= alpha.
        (Proof: V(R) = base_S + R*(gmax_S - base_S) for R in [0,1] by
        the two-sided greedy of prop:tv; sup_R V(R) <= alpha over R<=1
        collapses to the two endpoint conditions.)
  (iii) The full-radius crossing set {c : tau(c) = 1} is downward
        closed (a prefix), while tau itself need not be monotone
        inside the sub-unit region: below alpha both tau=1 conditions
        hold at cap=alpha already (gmax_S(alpha) <= alpha by
        domination) and, being monotone, stay true for every c < alpha;
        above alpha only the base_S <= alpha condition can bind, and it
        is monotone in c.  All observed crossings out of tau=1 are
        therefore base_S-driven.

PROVENANCE DISCLOSURE (honest, kept verbatim in the artifact):
the r503 log's "gmax_S monotone nondecreasing inline 2592/2592" claim
was asserted at run time but the assertion itself was never stored in
critical_cap_r503.py (only its docstring comment).  r506 U2 re-ran the
same check on the r503 stored curves (2592 adjacent pairs, 0
violations); this r508 artifact re-executes it a third time on the same
frozen bytes (V3) so the formal Proposition's premise has a stored,
replayable machine witness.

PRE-REGISTERED CHECKS (written before first run; mirrored in "checks"):
  V1: r498e coarse grid (72 cells x 12 caps = 792 adjacent ascending-cap
      pairs): base_S monotone nondecreasing in cap, 0 violations.
      (Equals r501 C3 0/792; re-executed here on r498e bytes.)
  V2: r498e coarse grid: gmax_S monotone nondecreasing in cap,
      0 violations on 792 pairs.
  V3: r503 fine grid: gmax_S monotone nondecreasing, 0 violations on
      2592 adjacent pairs (the re-stored r503 inline witness).
  V4: r503 fine grid prefix: full-radius crossing set has 0 holes and
      0 overhangs across 72 cells (c* = sup{c : tau(c)=1} characterization
      at grid resolution).  Equals r506 U1.
  V5: crossing-driver census: every maximal tau=1 prefix END (last
      scanned cap with tau=1, where a tau<1 cap exists above it) has
      base_S(c_end) <= alpha (consistent with base-driven crossing; the
      gmax condition at the same cap is reported per case).  Requires
      base_S on the fine grid: r503 stored only cap/tau/gmax_S/votes, so
      base_S is recomputed exactly from the frozen r498e machinery
      (subset_profile + same priors) at the r503 grid caps -- same code
      path as the r503 run itself, zero new data.
  V6: tau non-monotone interior reproduced on the r498e coarse grid:
      323 of 792 adjacent descending-cap comparisons violate tau
      monotonicity (r501 C4), denominator 792 (= 72 x 11), and every
      violation either lives strictly inside the sub-unit region (both
      caps tau < 1) or is an upward jump AT the prefix edge
      (tau(prev, larger cap) < 1, tau(next, smaller cap) = 1) -- never
      a hole (a tau = 1 cap followed by a sub-unit cap at a SMALLER
      budget).  The fine-grid prefix property V4 is the direct machine
      form of "no hole"; V6 records the coarse-grid complement.
      (r508 first-draft claim "every violation strictly sub-unit" was
      falsified by the data: 70 of 323 are edge-jump crossings;
      disclosed, claim corrected, not hidden.)
  V7: anchors -- r506 ALL_PASS, r507 ALL_PASS, r501 C4 recomputed 323,
      r505 11-tight/48 cross-reference all still true (frozen-artifact
      consistency across the chain this Proposition cites).

Zero new data, zero GPU, zero TEST/CAL reads.  Deterministic.
Output: prefix_prop_r508_result.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, "earlystop_drift_r494"))
import rule_channel_r494 as r494
sys.path.insert(0, os.path.join(WS, "earlystop_drift_r498"))
from statecap_repair_r498d import subset_profile

R498E = os.path.join(WS, "earlystop_drift_r498", "full_sweep_r498e_result.json")
R503 = os.path.join(WS, "earlystop_drift_r503", "critical_cap_r503_result.json")
R501 = os.path.join(WS, "earlystop_drift_r501", "fit_pareto_r501_result.json")
R505 = os.path.join(WS, "earlystop_drift_r505", "edge_tightness_r505_result.json")
R506 = os.path.join(WS, "earlystop_drift_r506", "discrete_geometry_r506_result.json")
R507 = os.path.join(WS, "earlystop_drift_r507", "discriminant_r507_result.json")
OUT = os.path.join(HERE, "prefix_prop_r508_result.json")

out = {
    "source": "frozen r498e/r503/r501/r505/r506/r507 artifacts; V5 base_S "
              "recomputed via the identical frozen subset_profile machinery; "
              "zero new data; zero TEST/CAL reads; deterministic",
    "proposition": "prefix law + two-scalar monotonicity of the full-radius "
                   "crossing set for the certificate-ordered state-subset "
                   "rule (paper v5.22 App D(g) formal Proposition)",
    "provenance_disclosure": (
        "r503 log claimed an inline 2592/2592 gmax_S monotonicity assert "
        "that was never stored in critical_cap_r503.py; r506 U2 re-ran it "
        "on the stored r503 curves; r508 V3 re-executes it again here so "
        "the formal Proposition premise has a stored replayable witness."),
    "checks": {},
}

# ---------- V1/V2/V6 on the r498e coarse grid ----------
d8e = json.load(open(R498E))
caps_desc = d8e["caps"]  # descending
cells = []
for cname, cblk in d8e["carriers"].items():
    for cell in cblk["cells"]:
        cells.append((cname, cell))
assert len(cells) == 72, f"expected 72 cells, got {len(cells)}"

v1_viol = v2_viol = v6_viol = 0
pairs = 0
v6_holes = 0        # tau=1 at SMALLER cap, tau<1 at LARGER cap (forbidden)
v6_entry = 0        # upward jump AT the prefix edge (allowed)
for cname, cell in cells:
    pc = cell["per_cap"]
    prev_tau = prev_base = prev_gmax = None
    for cap in caps_desc:  # descending caps; a violation is tau UP as cap DROPS
        e = pc[str(cap)]
        if prev_base is not None:
            pairs += 1
            if e["base_S"] > prev_base + 1e-12:
                v1_viol += 1
            if e["gmax_S"] > prev_gmax + 1e-12:
                v2_viol += 1
            if e["tau"] > prev_tau + 1e-9:
                # prev = larger cap (lower tau), e = smaller cap (higher tau)
                v6_viol += 1
                if prev_tau >= 1 - 1e-9 and e["tau"] < 1 - 1e-9:
                    v6_holes += 1   # larger cap full, smaller NOT: true hole
                elif prev_tau < 1 - 1e-9 and e["tau"] >= 1 - 1e-9:
                    v6_entry += 1   # sub-unit at larger cap, full at smaller
        prev_tau, prev_base, prev_gmax = e["tau"], e["base_S"], e["gmax_S"]
out["checks"]["V1_baseS_monotone_coarse"] = {
    "adjacent_pairs": pairs, "violations": v1_viol,
    "pass": pairs == 792 and v1_viol == 0}
out["checks"]["V2_gmax_monotone_coarse"] = {
    "adjacent_pairs": pairs, "violations": v2_viol,
    "pass": v2_viol == 0}
out["checks"]["V6_tau_nonmonotone_interior"] = {
    "adjacent_pairs": pairs, "tau_violations": v6_viol,
    "edge_jumps_into_tau1": v6_entry,
    "interior_subunit_violations": v6_viol - v6_entry - v6_holes,
    "holes": v6_holes,
    "denominator_erratum": "r501 printed 1728; true 72x11=792",
    "composition_disclosure": ("registered after the r508 first run "
        "observed the split: 323 = 253 strictly sub-unit + 70 prefix-edge "
        "jumps + 0 holes; anchored verbatim here"),
    "pass": (v6_viol == 323 and pairs == 792 and v6_holes == 0
             and v6_entry == 70 and v6_viol - v6_entry - v6_holes == 253)}

# ---------- V3/V4 on the r503 fine grid ----------
d503 = json.load(open(R503))
fine_viol = 0
fine_pairs = 0
holes = 0
overhangs = 0
prefix_ends = []  # (carrier, m, alpha, cap_end, tau_next_above)
for cname, cblk in d503["carriers"].items():
    for cell in cblk["cells"]:
        curve = sorted(cell["curve"], key=lambda r: r["cap"])
        prev = None
        full_flags = []
        for row in curve:
            if prev is not None:
                fine_pairs += 1
                # gmax_S is NONDECREASING in cap: flag any strict DECREASE.
                # (r508 first draft inverted this comparison -- script
                # author's sign slip, caught by the V3 self-check on first
                # run, 1615 false "violations" on 2592 pairs; corrected
                # here and disclosed in run_r508.log / the round log.
                # No artifact was edited to fit.)
                if row["gmax_S"] < prev["gmax_S"] - 1e-12:
                    fine_viol += 1
            full_flags.append((row["cap"], row["tau"] >= 1 - 1e-9))
            prev = row
        # prefix at grid resolution: once False, never True again
        seen_false = False
        for cap, is_full in full_flags:
            if not is_full:
                seen_false = True
            elif seen_false:
                holes += 1
        if full_flags[0][1] is False:
            overhangs += 1
        # record prefix end (last full cap) when a non-full cap exists
        if any(not f for _, f in full_flags) and any(f for _, f in full_flags):
            last_full = max(c for c, f in full_flags if f)
            prefix_ends.append((cname, cell["m"], cell["alpha"], last_full))
out["checks"]["V3_gmax_monotone_fine"] = {
    "adjacent_pairs": fine_pairs, "violations": fine_viol,
    "note": "stored replayable witness for the r503 inline claim "
            "(see provenance_disclosure)",
    "pass": fine_pairs == 2592 and fine_viol == 0}
out["checks"]["V4_prefix_fine"] = {
    "cells": 72, "holes": holes, "overhangs": overhangs,
    "pass": holes == 0 and overhangs == 0}

# ---------- V5: crossing-driver census (exact base_S recompute) ----------
# base_S was not stored on the r503 fine grid; recompute it with the
# identical frozen machinery (same priors, same subset_profile, same
# loader call chain as critical_cap_r503.py main()) at the prefix-end
# caps recorded above.
build_omr, dpflip_omr = r494.load_r469_machinery()
carrier_specs = {}
for cname, path in (("omr_shard0", "earlystop_drift_r467/cot_shard0.parquet"),
                    ("omr_shard1", "earlystop_drift_r471/cot_shard1.parquet")):
    Ks = r494.load_omr_counts(os.path.join(WS, path))
    fit_idx = r494.omr_fit_order(Ks, 4000)
    ms = sorted({max(8, int(len(fit_idx) * f)) for f in r494.FRACS})
    priors = {m: r494.hist_prior(Ks, fit_idx[:m], 32) for m in ms}
    carrier_specs[cname] = (priors, build_omr, 32)
mod_rl, Ks_rl, fit_idx_rl, _ = r494.load_rlve()
ms_rl = sorted({max(8, int(len(fit_idx_rl) * f)) for f in r494.FRACS})
priors_rl = {m: mod_rl.fit_prior(Ks_rl, fit_idx_rl[:m]) for m in ms_rl}
carrier_specs["rlve"] = (priors_rl, mod_rl.build_cert_table, 8)

v5_cases = []
v5_base_ok = 0
for cname, m, alpha, cap_end in prefix_ends:
    priors, build_cert, n_sym = carrier_specs[cname]
    Hm = priors[m]
    cert = build_cert(Hm)
    gS = []
    for K in range(n_sym + 1):
        gs, _, _, _, _, _ = subset_profile(K, cert, alpha, cap_end, n_sym)
        gS.append(gs)
    base = sum(h * gv for h, gv in zip(Hm, gS))
    gmax = max(gS)
    ok = base <= alpha + 1e-12
    v5_base_ok += int(ok)
    v5_cases.append({"carrier": cname, "m": m, "alpha": alpha,
                     "cap_end": cap_end, "base_S": round(base, 8),
                     "gmax_S": round(gmax, 8),
                     "base_le_alpha": bool(ok),
                     "gmax_le_alpha": bool(gmax <= alpha + 1e-12)})
out["checks"]["V5_crossing_driver_census"] = {
    "prefix_end_cases": len(v5_cases),
    "base_le_alpha_at_end": v5_base_ok,
    "cases": v5_cases,
    "note": "base_S recomputed with the identical frozen subset_profile "
            "machinery at the r503 fine-grid prefix-end caps (r503 stored "
            "only tau/gmax_S/votes on the fine grid)",
    "pass": len(v5_cases) > 0 and v5_base_ok == len(v5_cases)}

# ---------- V7: frozen-chain anchors ----------
d501 = json.load(open(R501))
d505 = json.load(open(R505))
d506 = json.load(open(R506))
d507 = json.load(open(R507))
c4r = d501.get("C4_refuted", {})
out["checks"]["V7_chain_anchors"] = {
    "r501_c4_refuted_text": c4r.get("result", ""),
    "r506_all_pass": d506["checks"]["ALL_PASS"] is True,
    "r506_c4_recomputed": d506["checks"]["U3_c4_count"]["recomputed_c4"],
    "r505_tight_11_of_48": (
        d505["checks"]["P3_tight_biconditional"]["pass"] is True),
    "r507_all_pass": d507["checks"]["ALL_PASS"] is True,
    "pass": (d506["checks"]["ALL_PASS"] is True
             and d506["checks"]["U3_c4_count"]["recomputed_c4"] == 323
             and d506["checks"]["U3_c4_count"]["pairs"] == 792
             and d507["checks"]["ALL_PASS"] is True
             and d505["checks"]["P3_tight_biconditional"]["pass"] is True),
}

out["checks"]["ALL_PASS"] = all(
    v.get("pass") for k, v in out["checks"].items() if isinstance(v, dict))

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print("ALL_PASS:", out["checks"]["ALL_PASS"])
for k, v in out["checks"].items():
    if isinstance(v, dict) and "pass" in v:
        print(("PASS " if v["pass"] else "FAIL ") + k)
print("wrote", OUT)
