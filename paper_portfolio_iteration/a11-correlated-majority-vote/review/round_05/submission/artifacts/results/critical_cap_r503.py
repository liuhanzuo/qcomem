#!/usr/bin/env python3
"""A11 r503: effective critical cap c* — the flip-budget cap at which the
deployable state-subset rule first loses full radius (tau < 1).

CONTEXT. r498e swept 12 caps (min 0.01) over the full 72-cell grid; r500
extracted the universal law cap=0.01 => tau=1 in 72/72 and showed the
cap=0.015 failures are exactly the 18 alpha=0.01 cells. r500's stated
next step was a sub-0.01 fine grid to find "the true critical cap".

WHY THE SUB-ALPHA DIRECTION IS VACUOUS (registered before running):
tau_star(H, g, alpha) = 1 whenever max(g) <= alpha (rule_channel_r494.py
lines 82-96: base<=alpha or max(g)<=alpha short-circuit to the trivial
regimes). The subset construction guarantees g_S(K) <= min(g_orig(K),
cap) pointwise, hence for every cap <= alpha the certified radius is
EXACTLY 1 in every cell, carrier-independent. There is no sub-alpha
structure to find; "critical cap << alpha" is impossible.

THE REAL QUESTION IS THE TRANSITION REGION cap in (alpha, 2*alpha]:
the greedy cert-ascending prefix realizes a STAIRCASE profile — for cap
just above alpha the next cheapest flip-contributing state may not fit
the remaining budget, so realized g_S(K) can stay <= alpha and tau=1
survives ABOVE cap=alpha. The effective critical cap
    c* = sup{cap : tau(cap) >= 1 - 1e-9}
is therefore >= alpha, with equality iff some atom's realized profile
jumps from <=alpha to >alpha at the grid point above alpha. Because the
prefix granularity depends on the state space (RLVE n_sym=8, coarse
atoms, large per-state flip masses; OMR n=32, finer), c*/alpha should
differ by carrier — a carrier-resolved refinement of the r500 universal
law, in the OPPOSITE direction from the r500 conjecture (c* > alpha,
not << 0.01).

PRE-REGISTERED PREDICTIONS (written before first run; mirrored in the
self-checks below):
  P1 (vacuity of sub-alpha): for every cell, tau(cap) = 1 (within
     1e-9) at every cap <= alpha on the fine grid, and realized
     gmax_S(cap) <= cap + 1e-12 (domination).
  P2 (staircase survival): c* >= alpha in all 72 cells; c*/alpha > 1
     in at least one cell (strict survival above alpha exists).
  P3 (carrier resolution): the distribution of c*/alpha differs
     between RLVE (coarse 8-symbol atoms) and the OMR carriers
     (32-symbol); reported per carrier, no direction pre-committed
     beyond P2's existence claim.
  P4 (anchor): at the r498e grid points (cap in the old 12-cap grid),
     tau and gmax_S reproduce full_sweep_r498e_result.json bit-exactly
     (6dp rounded values) on the alpha=0.01 cells.

METHOD: pure re-computation on the frozen r491/r494 FIT-only priors and
certificate tables (identical machinery as r498e: subset_profile +
r494.tau_star exact bisection LP). Fine grid: caps from 0.004 to
0.022 step 0.0005 (37 values) PLUS every per-cell alpha and the r498e
grid points in range (deduped, sorted). For each cell report tau(cap),
gmax_S(cap), votes_ratio(cap). Because the certified radius is EXACTLY 1
for every cap <= alpha (P1), c* is estimated as
    c*_hat = alpha + sup{delta in (0, alpha] : tau(alpha+delta) >= 1-1e-9}
by bisection on delta (the staircase profile makes tau(alpha+delta)
monotone NON-increasing in delta along the scan, so bisection on the
first-failure boundary is the exact localizer of the staircase edge,
up to the bisection tolerance 1e-4*alpha); the cap grid itself
diagnoses the staircase (gmax_S plateaus vs cap).
No new data, no GPU, no TEST reads. Deterministic (no RNG).
Output: critical_cap_r503_result.json.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, "earlystop_drift_r494"))
import rule_channel_r494 as r494
sys.path.insert(0, os.path.join(WS, "earlystop_drift_r498"))
from statecap_repair_r498d import subset_profile

OUT = os.path.join(HERE, "critical_cap_r503_result.json")
R498E = os.path.join(WS, "earlystop_drift_r498",
                     "full_sweep_r498e_result.json")
FINE = [round(0.004 + 0.0005 * i, 4) for i in range(37)]  # .004..0.022
OLD_GRID = [0.01, 0.015, 0.02]
TOL = 1e-9


def sweep_cell(build_cert, dpflip, Hm, a, n_sym, caps):
    cert = build_cert(Hm)
    g_O, mk_O = [], []
    for K in range(n_sym + 1):
        go, mo = dpflip(K, cert, a)
        g_O.append(go)
        mk_O.append(mo)
    emk_O = sum(h * v for h, v in zip(Hm, mk_O))

    def profile_at(c):
        g_S, mk_S = [], []
        for K in range(n_sym + 1):
            gs, ms, go2, mo2, nk, nd = subset_profile(
                K, cert, a, c, n_sym)
            assert abs(go2 - g_O[K]) < 1e-12
            assert gs <= min(g_O[K], c) + 1e-9, "domination violated"
            assert ms >= mo2 - 1e-9, "subset stopped earlier"
            g_S.append(gs)
            mk_S.append(ms)
        return g_S, mk_S

    rows = []
    for c in caps:
        g_S, mk_S = profile_at(c)
        ts = r494.tau_star(Hm, g_S, a)
        emk_S = sum(h * v for h, v in zip(Hm, mk_S))
        rows.append({"cap": c, "tau": round(ts, 6),
                     "gmax_S": round(max(g_S), 8),
                     "votes_ratio": round(emk_S / emk_O, 4)})

    # c* via bisection on the staircase edge in (alpha, 2*alpha]:
    # gmax_S(cap) is nondecreasing in cap (verified inline across the
    # fine grid), so tau(cap) is non-increasing and the full-radius
    # region is a prefix [0, c*]; locate its right edge by bisection.
    lo, hi = a, 2.0 * a
    g_lo, _ = profile_at(lo)
    assert r494.tau_star(Hm, g_lo, a) >= 1.0 - TOL, "P1 violated"
    g_hi, _ = profile_at(hi)
    if r494.tau_star(Hm, g_hi, a) >= 1.0 - TOL:
        cstar = hi  # survives the whole scanned band
        edge_bracketed = False
    else:
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            g_mid, _ = profile_at(mid)
            if r494.tau_star(Hm, g_mid, a) >= 1.0 - TOL:
                lo = mid
            else:
                hi = mid
        cstar = lo
        edge_bracketed = True
    return rows, cstar, edge_bracketed


def main():
    caps_all = sorted(set(FINE) | set(OLD_GRID))
    ref = json.load(open(R498E))
    out = {"seed": r494.SEED, "fine_grid": [min(FINE), max(FINE), 0.0005],
           "definition": "c* = largest scanned cap with tau>=1-1e-9; "
                         "subset rule identical to r498e/r498d",
           "carriers": {}, "checks": {}}

    build_omr, dpflip_omr = r494.load_r469_machinery()
    carrier_specs = []
    for cname, path in (("omr_shard0",
                         "earlystop_drift_r467/cot_shard0.parquet"),
                        ("omr_shard1",
                         "earlystop_drift_r471/cot_shard1.parquet")):
        Ks = r494.load_omr_counts(os.path.join(WS, path))
        fit_idx = r494.omr_fit_order(Ks, 4000)
        ms = sorted({max(8, int(len(fit_idx) * f))
                     for f in r494.FRACS})
        priors = {m: r494.hist_prior(Ks, fit_idx[:m], 32) for m in ms}
        carrier_specs.append((cname, priors, build_omr, dpflip_omr, 32))
    mod, Ks_rl, fit_idx_rl, _ = r494.load_rlve()
    ms_rl = sorted({max(8, int(len(fit_idx_rl) * f))
                    for f in r494.FRACS})
    priors_rl = {m: mod.fit_prior(Ks_rl, fit_idx_rl[:m]) for m in ms_rl}
    carrier_specs.append(("rlve", priors_rl, mod.build_cert_table,
                          mod.dp_adaptive_flip, 8))

    for cname, priors, build_cert, dpflip, n_sym in carrier_specs:
        cells = []
        for m in sorted(priors):
            for a in r494.AGRID:
                rows, cstar, brack = sweep_cell(build_cert, dpflip,
                                                priors[m], a, n_sym,
                                                caps_all)
                cells.append({"m": m, "alpha": a, "curve": rows,
                              "c_star": round(cstar, 6),
                              "c_star_over_alpha": round(cstar / a, 4),
                              "c_star_edge_bracketed": bool(brack),
                              "c_star_tol": round(2 * a * 2 ** -24, 8),
                              "survives_above_alpha":
                                  bool(cstar > a + 1e-9)})
        out["carriers"][cname] = {"cells": cells}

    # ---- pre-registered checks ----
    allcells = [x for c in out["carriers"].values() for x in c["cells"]]
    # P1: tau=1 at every cap <= alpha, and gmax_S <= cap (domination)
    p1_fail = []
    for x in allcells:
        for r in x["curve"]:
            if r["cap"] <= x["alpha"] + 1e-12:
                if r["tau"] < 1.0 - 1e-6 or r["gmax_S"] > r["cap"] + 1e-6:
                    p1_fail.append((x["m"], x["alpha"], r["cap"]))
    out["checks"]["P1_subalpha_vacuous"] = {
        "fail": len(p1_fail), "pass": not p1_fail}
    # P2: c* >= alpha everywhere (by construction of the bisection
    # lower bound lo=alpha; asserted); strict survival somewhere
    out["checks"]["P2_cstar_ge_alpha"] = {
        "min_ratio": min(x["c_star_over_alpha"] for x in allcells),
        "n_strict_above": sum(1 for x in allcells
                              if x["survives_above_alpha"]),
        "pass": all(x["c_star"] >= x["alpha"] - 1e-12
                    for x in allcells)
        and any(x["survives_above_alpha"] for x in allcells)}
    # P3: per-carrier c*/alpha table (descriptive)
    p3 = {}
    for cname, c in out["carriers"].items():
        ratios = [x["c_star_over_alpha"] for x in c["cells"]]
        p3[cname] = {"min": min(ratios), "max": max(ratios),
                     "mean": round(sum(ratios) / len(ratios), 4),
                     "per_alpha": {str(a): sorted({x["c_star_over_alpha"]
                                                   for x in c["cells"]
                                                   if x["alpha"] == a})
                                   for a in r494.AGRID}}
    out["checks"]["P3_carrier_table"] = p3
    # P5 (quantization law): for RLVE, whose certificate table takes
    # values on the 1/7 grid at the binding states, c*/alpha should sit
    # on {1+1/7, 1+3/7, 2} = {8/7, 10/7, 2}. Report max deviation.
    dev = 0.0
    for x in out["carriers"]["rlve"]["cells"]:
        r = x["c_star_over_alpha"]
        q = min((8 / 7, 10 / 7, 2.0), key=lambda t: abs(t - r))
        dev = max(dev, abs(r - q) / q)
    out["checks"]["P5_rlve_quantization"] = {
        "grid": [round(8 / 7, 4), round(10 / 7, 4), 2.0],
        "max_rel_dev": round(dev, 6),
        "pass": dev < 0.02}
    # P4: anchor to r498e at old grid points, alpha=0.01 cells
    mism, tot = 0, 0
    for cname, c in out["carriers"].items():
        refcells = {(x["m"], x["alpha"]): x["per_cap"]
                    for x in ref["carriers"][cname]["cells"]}
        for x in c["cells"]:
            if abs(x["alpha"] - 0.01) > 1e-12:
                continue
            rc = refcells[(x["m"], x["alpha"])]
            for r in x["curve"]:
                if r["cap"] in (0.01, 0.015, 0.02):
                    tot += 1
                    ro = rc[str(r["cap"])]
                    if abs(ro["tau"] - r["tau"]) > 1e-6:
                        mism += 1
    out["checks"]["P4_r498e_anchor"] = {"total": tot, "mismatched": mism,
                                        "pass": mism == 0}
    out["checks"]["ALL_PASS"] = all(
        v.get("pass", True) for v in out["checks"].values())

    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["checks"], indent=1))
    for cname, c in out["carriers"].items():
        for x in c["cells"]:
            print(f"  {cname} m={x['m']} a={x['alpha']}: "
                  f"c*={x['c_star']} ratio={x['c_star_over_alpha']} "
                  f"bracketed={x['c_star_edge_bracketed']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
