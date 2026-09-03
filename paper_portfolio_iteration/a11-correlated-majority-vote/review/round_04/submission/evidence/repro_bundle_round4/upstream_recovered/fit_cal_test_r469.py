#!/usr/bin/env python3
"""A11 r469 (MGR c8d5897e5f64 item ii+iii): problem-level FIT/CAL/TEST split,
legitimate simultaneous (k,alpha)-family selection on CAL, single TEST readout,
fair risk-cost comparison vs Hoeffding fixed-k baseline.

Design (pre-registered in this docstring):
  Split  : 11607 problems -> FIT 4000 / CAL 4000 / TEST 3607 (seeded shuffle).
  FIT    : estimate mixture prior Hhat (K-histogram) for BAYES-H. Rule frozen
           after FIT; CAL/TEST never touch the prior.
  CAL    : select rules from the family with simultaneous EB (Maurer-Pontil)
           UCBs + Bonferroni over the whole selection family:
             FIXED class : k in {3,5,...,31} x alpha in {0.10,0.05,0.02,0.01} (60 certs)
             BAYES-H     : alpha in same grid (4 certs, per-problem flip g_ad(K;alpha,Hhat)
                           computed EXACTLY by DP over (k,x) states, no MC)
           delta_cal = 0.05 total, per-test delta = 0.05/64.
           k*_EB(alpha) = min{k : EB-UCB_cal(k) <= alpha};
           BAYES-H(alpha) accepted iff EB-UCB_cal(g_ad) <= alpha.
           Hoeffding-fixed baseline: same grid, Hoeffding UCB, same delta budget.
  TEST   : ONE readout. Exact per-problem flip (hypergeom for FIXED; DP for
           BAYES-H) on the 3607 TEST problems. Report mean flip + Hoeffding CI
           (Bonferroni over reported rules), mean k + CI, rollout saving, and
           BAYES-H vs Hoeffding-fixed absolute/relative saving gap with paired CI.

Per-problem exact flip for FIXED k: f_K(k) = P_HG(MV(k) != MV(N) | K).
Per-problem exact flip for BAYES-H: DP over prefix states (k,x); stopping
criterion cert[k][x] = c_Hhat(x,k) <= alpha depends only on (k,x), so the
replay flip probability under true count K is an exact DP, not MC.

Self-check: DP adaptive flip vs r468-style replay MC on 200 problems
(assert agreement within 3*MC-sigma).

Readback: stdout + fit_cal_test_r469_result.json. stdlib+pandas/pyarrow. No GPU/net.
"""
import json, math, random
from math import comb, log, sqrt
import pandas as pd
import pyarrow.parquet as pq

PATH = "../earlystop_drift_r467/cot_shard0.parquet"
OUT = "fit_cal_test_r469_result.json"
N = 32
KGRID = list(range(3, N, 2))          # 15 odd k
AGRID = [0.10, 0.05, 0.02, 0.01]
DELTA_CAL = 0.05
J_FAM = len(KGRID) * len(AGRID) + len(AGRID)   # 60 + 4 = 64
D_TEST = DELTA_CAL / J_FAM
SEED = 20260815


def hyper_pmf(K, k, x):
    if x < max(0, k - (N - K)) or x > min(k, K):
        return 0.0
    return comb(K, x) * comb(N - K, k - x) / comb(N, k)


def side(cnt, k):
    return 1 if cnt > k / 2 else 0  # ties -> 0


# ---------- exact per-K tables ----------
def build_flip_fixed():
    """f[K][k] = P(prefix-k MV != full-N MV | count K), exact hypergeom sum."""
    f = {}
    for K in range(N + 1):
        full = side(K, N)
        row = {}
        for k in KGRID:
            row[k] = sum(hyper_pmf(K, k, x) for x in range(k + 1)
                         if side(x, k) != full)
        f[K] = row
    return f


def build_cert_table(H):
    """cert[k][x] = c_H(x,k) = posterior mass on K with side(K,N) != side(x,k)."""
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
    """Exact replay flip prob and mean stopping k of BAYES-H under count K.

    Returns (flip_prob, mean_k). stop at (k,x) iff k>=3 and cert[k][x]<=alpha,
    forced at k=N. Order = uniform random permutation of the K ones."""
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
            p1 = (K - x) / (N - k)   # next draw is a pass
            p0 = 1.0 - p1
            if p0 > 0:
                nxt[(k + 1, x)] = nxt.get((k + 1, x), 0.0) + p * p0
            if p1 > 0:
                nxt[(k + 1, x + 1)] = nxt.get((k + 1, x + 1), 0.0) + p * p1
        reach = nxt
    # forced stop at k=N
    for (kk, x), p in reach.items():
        assert kk == N
        ek += p * N
        if side(x, N) != full:
            flip += p
    return flip, ek


def eb_ucb(vals, delta):
    """Maurer-Pontil empirical Bernstein UCB for mean of [0,1] values."""
    m = len(vals)
    mu = sum(vals) / m
    var = sum((v - mu) ** 2 for v in vals) / (m - 1) if m > 1 else 0.0
    return mu + sqrt(2 * var * log(4 / delta) / m) + 7 * log(4 / delta) / (3 * (m - 1))


def hoef_ucb(vals, delta):
    m = len(vals)
    return sum(vals) / m + sqrt(log(1 / delta) / (2 * m))


def mean_ci(vals, delta):
    m = len(vals)
    mu = sum(vals) / m
    r = sqrt(log(2 / delta) / (2 * m))
    return mu, r


def main():
    t = pq.read_table(PATH, columns=["problem", "pass_rate_72b_tir"]).to_pandas()
    t = t[t.pass_rate_72b_tir.notna() & (t.pass_rate_72b_tir != "n/a")].copy()
    t = t.drop_duplicates(subset=["problem"])
    Ks = [int(round(p * N)) for p in t.pass_rate_72b_tir.astype(float).tolist()]
    n = len(Ks)

    rnd = random.Random(SEED)
    idx = list(range(n))
    rnd.shuffle(idx)
    fit_idx, cal_idx, test_idx = idx[:4000], idx[4000:8000], idx[8000:]

    # FIT prior
    H = [0.0] * (N + 1)
    for i in fit_idx:
        H[Ks[i]] += 1.0
    H = [h / len(fit_idx) for h in H]
    cert_tab = {a: build_cert_table(H) for a in AGRID}  # same H; alpha applied at stop

    f_fixed = build_flip_fixed()
    # exact adaptive flip/mean-k per K per alpha (DP; only depends on K)
    gad = {a: {K: dp_adaptive_flip(K, build_cert_table(H), a) for K in range(N + 1)}
           for a in AGRID}

    out = {"n": n, "n_fit": len(fit_idx), "n_cal": len(cal_idx), "n_test": len(test_idx),
           "delta_cal": DELTA_CAL, "J_family": J_FAM, "d_test": D_TEST,
           "seed": SEED, "N": N}

    # ---------- CAL selection ----------
    cal = {}
    for k in KGRID:
        cal[("F", k)] = [f_fixed[Ks[i]][k] for i in cal_idx]
    for a in AGRID:
        cal[("B", a)] = [gad[a][Ks[i]][0] for i in cal_idx]
        cal[("H", a)] = None  # placeholder, Hoeffding uses same fixed-k family

    sel = {}
    for a in AGRID:
        row = {}
        # FIXED-EB
        best = None
        for k in KGRID:
            u = eb_ucb(cal[("F", k)], D_TEST)
            if u <= a:
                best = k
                break
        row["FIXED_EB_k"] = best
        row["FIXED_EB_cert"] = (round(eb_ucb(cal[("F", best)], D_TEST), 5) if best else None)
        # FIXED-Hoeffding baseline
        bestH = None
        for k in KGRID:
            u = hoef_ucb(cal[("F", k)], D_TEST)
            if u <= a:
                bestH = k
                break
        row["FIXED_HOEF_k"] = bestH
        row["FIXED_HOEF_cert"] = (round(hoef_ucb(cal[("F", bestH)], D_TEST), 5) if bestH else None)
        # BAYES-H
        ub = eb_ucb(cal[("B", a)], D_TEST)
        row["BAYESH_cert"] = round(ub, 5)
        row["BAYESH_ok"] = bool(ub <= a)
        sel[str(a)] = row
    out["cal_selection"] = sel

    # ---------- self-check: DP vs replay MC on 200 problems ----------
    rnd2 = random.Random(7)
    cert05 = build_cert_table(H)
    maxdiff = 0.0
    R = 400
    for i in cal_idx[:200]:
        K = Ks[i]
        base = [1] * K + [0] * (N - K)
        fl = 0
        for _ in range(R):
            o = base[:]
            rnd2.shuffle(o)
            x = 0
            for k in range(1, N + 1):
                x += o[k - 1]
                if k >= 3 and cert05[k][x] <= 0.05:
                    fl += int(side(x, k) != side(K, N))
                    break
                if k == N:
                    fl += int(side(x, N) != side(K, N))
        mc = fl / R
        dp = gad[0.05][K][0]
        maxdiff = max(maxdiff, abs(mc - dp))
    tol = 3 * sqrt(0.25 / R)
    out["selfcheck_dp_vs_mc"] = {"max_abs_diff": round(maxdiff, 4), "tol_3sigma": round(tol, 4),
                                 "pass": bool(maxdiff <= tol)}
    assert maxdiff <= tol, "DP self-check failed"

    # ---------- single TEST readout ----------
    rules = []  # (name, kind, param, alpha)
    for a in AGRID:
        r = sel[str(a)]
        if r["FIXED_EB_k"]:
            rules.append((f"FIXED_EB_a{a}", "F", r["FIXED_EB_k"], a))
        if r["FIXED_HOEF_k"]:
            rules.append((f"FIXED_HOEF_a{a}", "F", r["FIXED_HOEF_k"], a))
        if r["BAYESH_ok"]:
            rules.append((f"BAYESH_a{a}", "B", a, a))
    rules.append(("FULL32", "FULL", N, None))
    n_rules = len(rules)
    d_rule = 0.05 / n_rules  # Bonferroni for descriptive TEST CIs

    test_res = {}
    bayesh_k_by_a = {}
    for name, kind, par, a in rules:
        flips, kss = [], []
        for i in test_idx:
            K = Ks[i]
            if kind == "F":
                flips.append(f_fixed[K][par])
                kss.append(float(par))
            elif kind == "B":
                fl, ek = gad[par][K]
                flips.append(fl)
                kss.append(ek)
            else:
                flips.append(0.0)
                kss.append(float(N))
        mu_f, r_f = mean_ci(flips, d_rule)
        mu_k, r_k = mean_ci(kss, d_rule)
        test_res[name] = {
            "alpha": a, "realized_flip": round(mu_f, 5), "flip_ci_radius": round(r_f, 5),
            "mean_k": round(mu_k, 3), "k_ci_radius": round(r_k, 4),
            "saving_vs_full": round(1 - mu_k / N, 4),
        }
        if kind == "B":
            bayesh_k_by_a[par] = kss
    out["test_readout"] = test_res
    out["test_ci_bonferroni_rules"] = n_rules

    # fair gap: BAYESH vs FIXED_HOEF at same alpha (paired per-problem k diff)
    gaps = {}
    for a in AGRID:
        r = sel[str(a)]
        if not (r["BAYESH_ok"] and r["FIXED_HOEF_k"]):
            continue
        kb = bayesh_k_by_a[a]
        kh = [float(r["FIXED_HOEF_k"])] * len(test_idx)
        diffs = [h - b for h, b in zip(kh, kb)]  # >0 means BAYES-H cheaper
        mu_d, r_d = mean_ci([d / N for d in diffs], d_rule)
        gaps[str(a)] = {
            "bayesh_saving": test_res[f"BAYESH_a{a}"]["saving_vs_full"],
            "hoef_saving": test_res[f"FIXED_HOEF_a{a}"]["saving_vs_full"],
            "abs_gap_bayesh_minus_hoef": round(mu_d, 4),
            "gap_ci_radius": round(r_d, 4),
            "rel_gain_vs_hoef": round(mu_d / (1 - r["FIXED_HOEF_k"] / N), 4) if r["FIXED_HOEF_k"] < N else None,
            "significant": bool(mu_d - r_d > 0),
        }
    out["fair_gap_bayesh_vs_hoeffding"] = gaps

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(sel, indent=1))
    print(json.dumps(test_res, indent=1))
    print(json.dumps(gaps, indent=1))
    print("selfcheck:", out["selfcheck_dp_vs_mc"])


if __name__ == "__main__":
    main()
