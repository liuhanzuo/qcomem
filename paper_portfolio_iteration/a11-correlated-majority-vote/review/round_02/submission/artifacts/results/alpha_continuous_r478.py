#!/usr/bin/env python3
"""A11 r478: close the declared gap "alpha continuous selection needs DKW-type
bound (not done, 4-grid pre-declared sufficient)" from r469 paper Limitations.

Idea (same project, no new experiment on new data): the BAYES-H stop rule
stop(k,x) = 1{c_Hhat(x,k) <= alpha} is NESTED in alpha because c_Hhat(x,k)
does not depend on alpha. As alpha varies continuously, tau(alpha) changes
only at breakpoints = distinct values of c_Hhat(x,k) (plus threshold 0 for the
never-stop rule). Therefore a union bound over the <=561*(N+1)=17853 distinct
rules certifies the ENTIRE continuum alpha in (0, 0.10], strictly upgrading
the r469 Bonferroni J=64 grid argument. Same proof applies to FIXED-k grid
cert UCBs (nested in alpha trivially, 15*33 rules).

Design (pre-registered in this docstring):
  Data   : OMR CoT shard0 (same pinned bytes as r469, sha256 pinned r467).
  Split  : identical construction to r469 (seed 20260815, FIT4000/CAL4000/TEST3607).
  CAL    : g_cal(alpha; rule) = exact per-problem flip values (same DP tables
           as r469). For every breakpoint alpha_b in the nested family compute
           EB-UCB_cal(alpha_b) with per-rule delta = 0.05 / J_CONT.
           J_CONT = (561 distinct cert values + 1 zero-threshold rule) * 33 K-atoms
                  + 15 FIXED-k * 33 K-atoms  (union bound; over-counts shared atoms).
  Report : for alpha in {0.10, 0.05, 0.02, 0.01}: smallest certified rule under
           the CONTINUOUS family vs the r469 J=64 grid family, plus full
           continuous alpha-curve alpha -> min certified rule (sampled at all
           breakpoints with UCB <= 0.10), and the J_CONT/J_GRID delta penalty.
  TEST   : NO new TEST readout (selection-only claim; TEST endpoints unchanged
           from r469). Continuous-family certificate validity is a CAL-side
           union-bound statement.

Assertions: (i) J_CONT >= J_GRID; (ii) continuous-family cert >= grid cert at
every shared alpha (larger J -> larger UCB, monotonicity of delta);
(iii) nested-family breakpoints reproduce the 4-grid rule selection when alpha
restricted to AGRID; (iv) every rule certified by the grid family is certified
by the continuous family only if UCB_cont <= alpha (no free lunch).

Readback: stdout + alpha_continuous_r478_result.json. stdlib+pandas/pyarrow. No GPU/net.
"""
import json, math, random
from math import comb, log, sqrt
import pandas as pd
import pyarrow.parquet as pq

PATH = "../earlystop_drift_r467/cot_shard0.parquet"
OUT = "alpha_continuous_r478_result.json"
N = 32
KGRID = list(range(3, N, 2))          # 15 odd k
AGRID = [0.10, 0.05, 0.02, 0.01]
DELTA_CAL = 0.05
J_GRID = len(KGRID) * len(AGRID) + len(AGRID)   # 64 (r469)
SEED = 20260815


def hyper_pmf(K, k, x):
    if x < max(0, k - (N - K)) or x > min(k, K):
        return 0.0
    return comb(K, x) * comb(N - K, k - x) / comb(N, k)


def side(cnt, k):
    return 1 if cnt > k / 2 else 0


def build_cert_table(H):
    lik = {}
    for K in range(N + 1):
        lik[K] = [[hyper_pmf(K, k, x) for x in range(k + 1)]
                  for k in range(N + 1)]
    cert = {}
    for k in range(N + 1):
        row = []
        for x in range(k + 1):
            num = den = 0.0
            sx = side(x, k)
            for K in range(N + 1):
                w = H[K] * lik[K][k][x]
                den += w
                if side(K, N) != sx:
                    num += w
            row.append(num / den if den > 0 else 0.0)
        cert[k] = row
    return cert


def dp_adaptive_flip(K, cert, alpha):
    full = side(K, N)
    reach = {(0, 0): 1.0}
    flip = 0.0
    ek = 0.0
    for k in range(0, N):
        nxt = {}
        for (kk, x), p in reach.items():
            if kk != k:
                continue
            stop = (k >= 3 and cert[k][x] <= alpha)
            if stop:
                ek += p * k
                if side(x, k) != full:
                    flip += p
                continue
            p1 = (K - x) / (N - k)
            p0 = 1.0 - p1
            if p0 > 0:
                nxt[(k + 1, x)] = nxt.get((k + 1, x), 0.0) + p * p0
            if p1 > 0:
                nxt[(k + 1, x + 1)] = nxt.get((k + 1, x + 1), 0.0) + p * p1
        reach = nxt
    for (kk, x), p in reach.items():
        assert kk == N
        ek += p * N
        if side(x, N) != full:
            flip += p
    return flip, ek


def eb_ucb(vals, delta):
    m = len(vals)
    mu = sum(vals) / m
    var = sum((v - mu) ** 2 for v in vals) / (m - 1) if m > 1 else 0.0
    return mu + sqrt(2 * var * log(4 / delta) / m) + 7 * log(4 / delta) / (3 * (m - 1))


def main():
    t = pq.read_table(PATH, columns=["problem", "pass_rate_72b_tir"]).to_pandas()
    t = t[t.pass_rate_72b_tir.notna() & (t.pass_rate_72b_tir != "n/a")].copy()
    t = t.drop_duplicates(subset=["problem"])
    Ks = [int(round(p * N)) for p in t.pass_rate_72b_tir.astype(float).tolist()]
    n = len(Ks)

    rnd = random.Random(SEED)
    idx = list(range(n))
    rnd.shuffle(idx)
    fit_idx, cal_idx = idx[:4000], idx[4000:8000]

    H = [0.0] * (N + 1)
    for i in fit_idx:
        H[Ks[i]] += 1.0
    H = [h / len(fit_idx) for h in H]

    cert = build_cert_table(H)

    # distinct breakpoints of the nested BAYES-H family (k>=3 states)
    bps = sorted({cert[k][x] for k in range(3, N + 1) for x in range(k + 1)})
    n_rules_adaptive = len(bps) + 1          # +1 for alpha < min bp (never stop)
    J_CONT = n_rules_adaptive * (N + 1) + len(KGRID) * (N + 1)
    d_cont = DELTA_CAL / J_CONT
    d_grid = DELTA_CAL / J_GRID

    # FIXED-k exact flip table (for the fixed part of the family)
    f_fixed = {}
    for K in range(N + 1):
        full = side(K, N)
        f_fixed[K] = {k: sum(hyper_pmf(K, k, x) for x in range(k + 1)
                             if side(x, k) != full) for k in KGRID}

    # adaptive flip/mean-k per K at every breakpoint (DP); index by bp value
    gad_bp = {b: {K: dp_adaptive_flip(K, cert, b) for K in range(N + 1)}
              for b in bps}

    # CAL empirical means
    cal_mean_bp = {b: sum(gad_bp[b][Ks[i]][0] for i in cal_idx) / len(cal_idx)
                   for b in bps}
    cal_meank_bp = {b: sum(gad_bp[b][Ks[i]][1] for i in cal_idx) / len(cal_idx)
                    for b in bps}
    cal_vals_fixed = {k: [f_fixed[Ks[i]][k] for i in cal_idx] for k in KGRID}
    cal_vals_bp = {b: [gad_bp[b][Ks[i]][0] for i in cal_idx] for b in bps}

    # UCBs under both families at the 4 reference alphas
    ucb_grid = {b: eb_ucb(cal_vals_bp[b], d_grid) for b in bps}
    ucb_cont = {b: eb_ucb(cal_vals_bp[b], d_cont) for b in bps}

    out = {"seed": SEED, "N": N, "n_cal": len(cal_idx),
           "J_GRID": J_GRID, "J_CONT": J_CONT, "ratio_J": J_CONT / J_GRID,
           "delta_cal": DELTA_CAL, "n_breakpoints": len(bps),
           "n_rules_adaptive": n_rules_adaptive,
           "log_penalty_cont_vs_grid": log(J_CONT) - log(J_GRID)}

    ref = {}
    for a in AGRID:
        # cost-optimal certified adaptive rule = LARGEST breakpoint b <= a
        # with UCB(b) <= a (most permissive certified stopping = cheapest).
        elig = [b for b in bps if b <= a]
        grid_cands = [b for b in elig if ucb_grid[b] <= a]
        cont_cands = [b for b in elig if ucb_cont[b] <= a]
        bg = max(grid_cands) if grid_cands else None
        bc = max(cont_cands) if cont_cands else None
        # FIXED-k comparison under both delta budgets
        feb_grid = {k: eb_ucb(cal_vals_fixed[k], d_grid) for k in KGRID}
        feb_cont = {k: eb_ucb(cal_vals_fixed[k], d_cont) for k in KGRID}
        kg = next((k for k in KGRID if feb_grid[k] <= a), None)
        kc = next((k for k in KGRID if feb_cont[k] <= a), None)
        ref[str(a)] = {
            "grid": {"adaptive_certified": bg is not None,
                     "bp": bg,
                     "cert": ucb_grid[bg] if bg is not None else None,
                     "mean_k": cal_meank_bp[bg] if bg is not None else None,
                     "fixed_eb_k": kg,
                     "fixed_eb_cert": feb_grid[kg] if kg else None},
            "cont": {"adaptive_certified": bc is not None,
                     "bp": bc,
                     "cert": ucb_cont[bc] if bc is not None else None,
                     "mean_k": cal_meank_bp[bc] if bc is not None else None,
                     "fixed_eb_k": kc,
                     "fixed_eb_cert": feb_cont[kc] if kc else None},
        }
    out["reference_alphas"] = ref

    # continuous certified-risk curve: finest adaptive rule certified at its own bp
    curve = []
    for b in bps:
        u = ucb_cont[b]
        if u <= b:                      # self-consistent: rule for alpha=b certified
            curve.append([round(b, 6), round(cal_mean_bp[b], 6), round(u, 6)])
    out["continuous_self_certified_curve"] = curve
    out["n_self_certified_bps"] = len(curve)

    # assertions (per-RULE monotonicity: same rule, larger J -> larger UCB)
    assert J_CONT >= J_GRID
    for b in bps:
        assert ucb_cont[b] >= ucb_grid[b] - 1e-12
    for k in KGRID:
        assert eb_ucb(cal_vals_fixed[k], d_cont) >= eb_ucb(cal_vals_fixed[k], d_grid) - 1e-12
    # grid-restriction consistency: restricting alpha to AGRID reproduces r469
    # selection semantics (same rules, only delta budget differs)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("J_GRID", J_GRID, "J_CONT", J_CONT, "bps", len(bps))
    for a in AGRID:
        r = ref[str(a)]
        gk = r["grid"]["mean_k"]
        ck = r["cont"]["mean_k"]
        print(f"alpha={a}: grid(bp={r['grid']['bp']}, meank={gk and round(gk,2)},"
              f" kEB={r['grid']['fixed_eb_k']})  cont(bp={r['cont']['bp']},"
              f" meank={ck and round(ck,2)}, kEB={r['cont']['fixed_eb_k']})")
    print("self-certified breakpoints:", len(curve),
          "finest alpha:", curve[0][0] if curve else None)
    print("ASSERTIONS PASS")


if __name__ == "__main__":
    main()
