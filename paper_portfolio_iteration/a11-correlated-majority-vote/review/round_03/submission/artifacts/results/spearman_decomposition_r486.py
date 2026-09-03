#!/usr/bin/env python3
"""A11 r486: adversarial decomposition audit of the pooled Spearman-0.75
alignment claim (paper v5.8 Sec.4, prop:tv cross-carrier paragraph +
App D(d)).

Motivation (peer audit imported): A1 r1108 showed a pooled correlation
(rho=-0.54, n=8, n.s.) can fail to predict onset while a per-unit
discriminating probe succeeds; A9 r263 showed pooled (convex-mixture)
quantities can be agnostic to dispersion and mislead necessity arguments.
Our claim "TV-robustness is driven by the alignment of the prior with the
zero-g set" rests on ONE pooled Spearman over 16 carrier x level cells
(r484). A hostile reviewer can recompute and ask: is this a within-carrier
law or a between-carrier artifact (2 of 4 carriers carry whole-simplex
tau*=1 plateaus, so a few cells may dominate)?

This audit decomposes the 16-cell statistic WITHOUT touching the paper's
claim yet: if the decomposition supports the claim we keep v5.8 wording;
if it refutes or weakens it we repair the wording in the same round
(continuous-work rule; no waiting).

Pre-registered decomposition (written BEFORE first run):
  Q1: pooled 16-cell Spearman reproduces 0.7544 (r484 JSON) bit-exactly
      (midrank tie-aware).
  Q2: within-carrier Spearman per carrier (4 levels each, n=4):
      PREDICTION (from r484 curve shapes, not yet computed): OMR shards
      zero-g mass is nearly CONSTANT across levels within a shard (the
      DP rule stops on already-decided states at all 4 levels, so the
      zero-g set barely moves) => within-OMR Spearman is unstable
      (near-zero x-variance); RLVE zero-g mass has ties (.02==.01) and
      tau* plateaus => degenerate; OpenR1 zero-g constant .768 across
      levels (support fixed at p=0,1) => x-variance EXACTLY zero,
      Spearman UNDEFINED. So within-carrier correlations are mostly
      degenerate by construction -- the pooled 0.75 CANNOT be a
      within-carrier law; it is a BETWEEN-carrier statement.
  Q3: between-carrier-only Spearman: collapse to 4 carrier-mean cells
      (mean zero-g, mean tau* per carrier): n=4, ties possible.
  Q4: carrier-removed sensitivity: recompute pooled Spearman dropping one
      carrier at a time (12 cells each). PREDICTION: dropping OpenR1
      (2 plateau cells with extreme tau*=1) and dropping RLVE (2 plateau
      cells) should move the statistic the most; if pooled stays ~0.7+
      under every single-carrier removal, the claim is not
      single-carrier-driven.
  Q5: plateaus-removed: recompute pooled Spearman on cells with tau*<1
      only (remove whole-simplex plateaus, which are degenerate
      "infinite robustness" cells).
  Q6: alternative driver check (dispersion/heterogeneity): Spearman of
      het-excess variance vs tau* across the same 16 cells, to verify the
      paper's negative clause "not by heterogeneity alone". OMR het-excess
      0.129 (both shards), RLVE 0.0999, OpenR1: compute from 3-atom prior.
      Within-carrier het is CONSTANT across levels (prior fixed per
      carrier), so this is again a between-carrier statement; we check
      its sign and magnitude.

Outputs: spearman_decomposition_r486_result.json + stdout readback.
Zero new data, zero TEST re-reads; reuses r484 frozen machinery via import.
"""
import json, os, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
OUT = os.path.join(HERE, "spearman_decomposition_r486_result.json")

# --- import the frozen r484 module (main() is __name__-guarded) ----------
spec = importlib.util.spec_from_file_location(
    "r484", os.path.join(WS, "earlystop_drift_r484", "tv_conservation_r484.py"))
r484 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r484)

LEVELS = [0.10, 0.05, 0.02, 0.01]


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and abs(v[order[j + 1]] - v[order[i]]) < 1e-12:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    dx, dy = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(dx) / n, sum(dy) / n
    vx = sum((a - mx) ** 2 for a in dx)
    vy = sum((b - my) ** 2 for b in dy)
    if vx < 1e-24 or vy < 1e-24:
        return None  # degenerate: zero variance in one coordinate
    num = sum((a - mx) * (b - my) for a, b in zip(dx, dy))
    return num / (vx * vy) ** 0.5


def het_excess(H):
    """heterogeneity excess variance matching the carrier artifacts
    (r467 0.12563 / r471 0.12544 / r474 0.09986): Var_p(H) minus the
    MEAN per-task binomial floor E_H[p(1-p)/N]. FIRST VERSION of this
    docstring used the pooled floor pbar(1-pbar)/N -- wrong (Jensen
    splits exactly the het-excess term into the mean floor); self-caught
    before any downstream number was written (0.120 != 0.126 artifact)."""
    n = len(H) - 1
    pbar = sum(K / n * h for K, h in enumerate(H))
    var = sum(h * (K / n - pbar) ** 2 for K, h in enumerate(H))
    base = sum(h * (K / n) * (1 - K / n) / n for K, h in enumerate(H))
    return var - base


def main():
    build, dpflip = r484.load_r469_machinery()

    carriers = {}  # name -> {H, cert-or-dp flip fn, N}
    for cname, path in (("omr_shard0", "earlystop_drift_r467/cot_shard0.parquet"),
                        ("omr_shard1", "earlystop_drift_r471/cot_shard1.parquet")):
        H = r484.fit_omr_prior(r484.load_omr_counts(os.path.join(WS, path)))
        cert = build(H)
        carriers[cname] = {
            "H": H,
            "g": lambda a, c=cert, d=dpflip: [d(K, c, a)[0] for K in range(33)],
        }
    mod, Hrl, cert_rl = r484.load_rlve()
    carriers["rlve"] = {
        "H": Hrl,
        "g": lambda a: [mod.dp_adaptive_flip(K, cert_rl, a)[0] for K in range(9)],
    }
    Ho = r484.openr1_prior()
    carriers["openr1"] = {"H": Ho, "g": r484.openr1_g}

    # build the 16 cells
    cells = []  # (carrier, alpha, zero_g, tau, het)
    for cname, c in carriers.items():
        H = c["H"]
        het = het_excess(H)
        for a in LEVELS:
            g = c["g"](a)
            zg = sum(h for h, gv in zip(H, g) if gv <= 1e-12)
            tau = r484.tau_star(H, g, a)
            cells.append({"carrier": cname, "alpha": a, "zero_g": zg,
                          "tau": tau, "het": het})

    out = {"seed": r484.SEED, "levels": LEVELS,
           "audit": "decomposition of the 16-cell pooled Spearman-0.75 "
                    "alignment claim; zero new data, zero TEST re-reads",
           "cells": cells}

    zg = [c["zero_g"] for c in cells]
    tau = [c["tau"] for c in cells]
    het = [c["het"] for c in cells]

    # Q1: reproduce r484 pooled
    pooled = spearman(zg, tau)
    r484ref = json.load(open(os.path.join(
        WS, "earlystop_drift_r484", "tv_conservation_r484_result.json")))
    out["q1_pooled_16"] = {"spearman": round(pooled, 4),
                           "r484_json": r484ref["zerog_tau_spearman_16cells"],
                           "match": abs(pooled - r484ref["zerog_tau_spearman_16cells"]) < 5e-5}

    # Q2: within-carrier (n=4 each)
    out["q2_within_carrier"] = {}
    for cname in carriers:
        sub = [c for c in cells if c["carrier"] == cname]
        s = spearman([c["zero_g"] for c in sub], [c["tau"] for c in sub])
        out["q2_within_carrier"][cname] = {
            "spearman": (None if s is None else round(s, 4)),
            "zero_g": [round(c["zero_g"], 4) for c in sub],
            "tau": [round(c["tau"], 4) for c in sub],
            "degenerate": s is None}

    # Q3: between-carrier means (n=4)
    means = {}
    for cname in carriers:
        sub = [c for c in cells if c["carrier"] == cname]
        means[cname] = {"zero_g": sum(c["zero_g"] for c in sub) / 4,
                        "tau": sum(c["tau"] for c in sub) / 4}
    sb = spearman([m["zero_g"] for m in means.values()],
                  [m["tau"] for m in means.values()])
    out["q3_between_carrier_means"] = {
        "spearman": (None if sb is None else round(sb, 4)),
        "means": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                  for k, v in means.items()}}

    # Q4: leave-one-carrier-out (12 cells)
    out["q4_loco"] = {}
    for drop in carriers:
        sub = [c for c in cells if c["carrier"] != drop]
        s = spearman([c["zero_g"] for c in sub], [c["tau"] for c in sub])
        out["q4_loco"][drop] = None if s is None else round(s, 4)

    # Q5: plateaus removed (tau*<1)
    sub = [c for c in cells if c["tau"] < 1 - 1e-12]
    s5 = spearman([c["zero_g"] for c in sub], [c["tau"] for c in sub])
    out["q5_plateaus_removed"] = {
        "n_cells": len(sub),
        "dropped": [f"{c['carrier']}@{c['alpha']}" for c in cells
                    if c["tau"] >= 1 - 1e-12],
        "spearman": None if s5 is None else round(s5, 4)}

    # Q6: heterogeneity as alternative driver (16 cells; constant within
    # carrier, so effectively between-carrier)
    s6 = spearman(het, tau)
    out["q6_het_vs_tau_16"] = {"spearman": None if s6 is None else round(s6, 4),
                               "het_by_carrier": {k: round(het_excess(c["H"]), 4)
                                                  for k, c in carriers.items()}}
    # and het vs zero-g across the 16 cells (are the two drivers confounded?)
    s7 = spearman(het, zg)
    out["q7_het_vs_zerog_16"] = None if s7 is None else round(s7, 4)

    # Q8: het-excess cross-check against pinned carrier artifacts
    # (r467 full-pool OMR shard0, r471 shard1 within, r474 RLVE within).
    # FIT-prior values differ from full-pool/within artifacts by split
    # sampling noise; require agreement to 0.01 (split-level tolerance).
    r467 = json.load(open(os.path.join(
        WS, "earlystop_drift_r467", "passrate_r467_result.json")))
    r471 = json.load(open(os.path.join(
        WS, "earlystop_drift_r471", "shard1_robust_r471_result.json")))
    r474 = json.load(open(os.path.join(
        WS, "earlystop_drift_r474", "rlve_n8_r474_result.json")))
    hets = {k: round(het_excess(c["H"]), 4) for k, c in carriers.items()}
    out["q8_het_artifact_xcheck"] = {
        "fit_prior_values": hets,
        "artifacts": {"omr_shard0_fullpool_r467": r467["het_excess_var"],
                      "omr_shard1_within_r471": r471["R1_within"]["het_excess_var"],
                      "rlve_within_r474": r474["within"]["het_excess_var"]},
        "match_within_0.01": all([
            abs(hets["omr_shard0"] - r467["het_excess_var"]) < 0.01,
            abs(hets["omr_shard1"] - r471["R1_within"]["het_excess_var"]) < 0.01,
            abs(hets["rlve"] - r474["within"]["het_excess_var"]) < 0.01])}

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"},
                     indent=1))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
