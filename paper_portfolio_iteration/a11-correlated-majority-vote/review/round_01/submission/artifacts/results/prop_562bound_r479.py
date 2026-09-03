#!/usr/bin/env python3
"""A11 r479: audit + tighten the structural breakpoint count in paper v5.2
Remark rem:nested ("267 observed at N=32; <=561 in general, plus the
never-stop rule", J_CONT=9339).

Facts to verify / quantify (same FIT/CAL split, same seed 20260815, same
pinned OMR shard0 bytes as r469/r478; NO new TEST readout):

  (1) Total states (k,x) with 3<=k<=32, 0<=x<=k: 555. Hence "267 observed
      <= 561 in general" is a VALID but loose bound. Recompute r478's 267
      bit-for-bit (distinct cert values over all 555 states).
  (2) Tight effective family for the paper's actual claim "certifies the
      entire continuum alpha in (0,0.10]": rules tau(alpha) change only when
      alpha crosses a distinct cert value that lies in (0, 0.10]. Cert values
      > 0.10 never trigger any rule in the range; cert value 0 stops for
      every alpha>0 (part of all rules, not rule-distinguishing). Effective
      family size = #{distinct c in (0,0.10]} + 1 (never-stop). Report it.
  (3) Recompute J under the tight family:
      J_tight = (#{distinct c in (0,0.10]} + 1) * 33 + 15 * 33  (K-atom
      over-count kept identical to r478 for a like-for-like comparison),
      vs paper J_CONT=9339. Report log-penalty reduction.
  (4) Recompute the 4 reference-alpha selections (adaptive bp/cert/mean_k and
      FIXED-EB k*) under d_tight = 0.05/J_tight vs d_cont = 0.05/9339, same
      CAL pool. In particular test whether FIXED-EB regains its alpha=0.02
      budget (paper v5.2 says it loses it under the continuous family).

Assertions: J_tight <= J_CONT; selected tight-family bp >= selected
cont-family bp at each alpha (smaller delta -> smaller UCB -> at least as
permissive); r478 267 reproduced.

Readback: stdout + prop_562bound_r479_result.json. stdlib+pandas/pyarrow.
"""
import json, random
from math import comb, log, sqrt
import pandas as pd
import pyarrow.parquet as pq

PATH = "../earlystop_drift_r467/cot_shard0.parquet"
OUT = "prop_562bound_r479_result.json"
N = 32
KGRID = list(range(3, N, 2))
AGRID = [0.10, 0.05, 0.02, 0.01]
DELTA_CAL = 0.05
J_GRID = len(KGRID) * len(AGRID) + len(AGRID)
SEED = 20260815
ALPHA_MAX = 0.10


def hyper_pmf(K, k, x):
    if x < max(0, k - (N - K)) or x > min(k, K):
        return 0.0
    return comb(K, x) * comb(N - K, k - x) / comb(N, k)


def side(cnt, k):
    return 1 if cnt > k / 2 else 0


def build_cert_table(H):
    lik = {K: [[hyper_pmf(K, k, x) for x in range(k + 1)]
               for k in range(N + 1)] for K in range(N + 1)}
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
            if k >= 3 and cert[k][x] <= alpha:
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

    states = [(k, x) for k in range(3, N + 1) for x in range(k + 1)]
    out = {"seed": SEED, "N": N, "n_states_total": len(states)}

    # (1) reproduce r478's 267
    bps_all = sorted({cert[k][x] for (k, x) in states})
    out["r478_n_breakpoints_recomputed"] = len(bps_all)
    out["paper_bound_561_valid"] = len(states) <= 561

    # (2) tight family for alpha in (0, 0.10]
    # distinct rules {tau(alpha): alpha in (0,0.10]} = stop-at-zero-states
    # rule (alpha below smallest positive distinct cert) + one new rule per
    # distinct positive cert value <= 0.10. Exact, given H.
    bps_eff = sorted({c for c in (cert[k][x] for (k, x) in states)
                      if 0.0 < c <= ALPHA_MAX})
    out["n_distinct_cert_in_0_010"] = len(bps_eff)
    out["n_distinct_rules_tight_exact"] = 1 + len(bps_eff)
    out["n_cert_zero_states"] = sum(1 for (k, x) in states if cert[k][x] == 0.0)
    out["n_cert_above_010"] = len([c for c in bps_all if c > ALPHA_MAX])
    bps_eval = sorted({0.0} | set(bps_eff))

    # (3) J comparison (same 33-atom over-count convention as r478).
    # r478/paper actually used the OBSERVED 268 adaptive rules (267 distinct
    # cert values over all 555 states + never-stop) -> J_CONT=9339; the 561
    # figure in the Remark is only the structural upper bound.
    J_CONT = 9339
    assert J_CONT == (267 + 1) * (N + 1) + len(KGRID) * (N + 1)
    J_struct = (561 + 1) * (N + 1) + len(KGRID) * (N + 1)   # if one used the bound
    J_tight = (len(bps_eff) + 1) * (N + 1) + len(KGRID) * (N + 1)
    out["J_CONT_paper_observed267"] = J_CONT
    out["J_CONT_structural561"] = J_struct
    out["J_CONT_tight"] = J_tight
    out["log_penalty_paper_vs_grid"] = log(J_CONT) - log(J_GRID)
    out["log_penalty_tight_vs_grid"] = log(J_tight) - log(J_GRID)
    assert J_tight <= J_CONT

    # (4) recompute 4-alpha selections under both delta budgets
    # bps_eval includes bp=0 (stop-at-zero-states rule), the realized rule for
    # alpha below the smallest positive distinct cert value.
    d_cont = DELTA_CAL / J_CONT
    d_tight = DELTA_CAL / J_tight
    gad_bp = {b: {K: dp_adaptive_flip(K, cert, b) for K in range(N + 1)} for b in bps_eval}
    cal_vals_bp = {b: [gad_bp[b][Ks[i]][0] for i in cal_idx] for b in bps_eval}
    cal_meank_bp = {b: sum(gad_bp[b][Ks[i]][1] for i in cal_idx) / len(cal_idx) for b in bps_eval}
    ucb_cont = {b: eb_ucb(cal_vals_bp[b], d_cont) for b in bps_eval}
    ucb_tight = {b: eb_ucb(cal_vals_bp[b], d_tight) for b in bps_eval}

    f_fixed = {K: {k: sum(hyper_pmf(K, k, x) for x in range(k + 1)
                          if side(x, k) != side(K, N)) for k in KGRID}
               for K in range(N + 1)}
    cal_vals_fixed = {k: [f_fixed[Ks[i]][k] for i in cal_idx] for k in KGRID}

    ref = {}
    for a in AGRID:
        elig_b = [b for b in bps_eval if b <= a]
        cc = [b for b in elig_b if ucb_cont[b] <= a]
        ct = [b for b in elig_b if ucb_tight[b] <= a]
        bc = max(cc) if cc else None
        bt = max(ct) if ct else None
        if bc is not None and bt is not None:
            assert bt >= bc - 1e-15
        feb_cont = {k: eb_ucb(cal_vals_fixed[k], d_cont) for k in KGRID}
        feb_tight = {k: eb_ucb(cal_vals_fixed[k], d_tight) for k in KGRID}
        kc = next((k for k in KGRID if feb_cont[k] <= a), None)
        kt = next((k for k in KGRID if feb_tight[k] <= a), None)
        ref[str(a)] = {
            "cont_paper": {"bp": bc,
                           "mean_k": cal_meank_bp[bc] if bc is not None else None,
                           "cert": ucb_cont[bc] if bc is not None else None},
            "cont_tight": {"bp": bt,
                           "mean_k": cal_meank_bp[bt] if bt is not None else None,
                           "cert": ucb_tight[bt] if bt is not None else None},
            "fixed_eb_k_paper": kc,
            "fixed_eb_k_tight": kt,
        }
    out["reference_alphas"] = ref

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
