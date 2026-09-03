#!/usr/bin/env python3
"""A11 r471: shard1 robustness — independent replication + prior-mismatch transfer.

Two questions, both pre-registered in this docstring:
  (R1) Independent replication: rerun the full FIT/CAL/TEST chain of r469 on
       OpenMathReasoning CoT shard1 (a disjoint set of problems). Do the
       qualitative conclusions survive? (BAYES-H saving dominance, EB-vs-Hoeffding
       selection gap, calibration at alpha in {0.10,0.05,0.02}.)
  (R2) Prior-mismatch transfer: fit the BAYES-H prior H on shard0 FIT, then run
       the CAL selection and TEST readout on shard1 problems. This is the
       deployment-realistic drift axis: the prior was estimated on a different
       problem pool. Hypothesis (from r468 robust_src): certificate stays
       conservative (realized flip <= alpha) because prior mismatch only inflates
       the posterior flip estimate in the conservative direction for U-shaped
       mixtures.

Method is identical to r469 (fit_cal_test_r469.py) — same split sizes, same
family (J=64), same delta budget, same EB/Hoeffding UCBs, same exact DP.
Difference: data source and the R2 prior provenance.

Readback: stdout + shard1_robust_r471_result.json. stdlib+pandas/pyarrow. No GPU/net.
"""
import json, math, random
from math import comb, log, sqrt
import pandas as pd
import pyarrow.parquet as pq

SHARD1 = "cot_shard1.parquet"
SHARD0 = "../earlystop_drift_r467/cot_shard0.parquet"
OUT = "shard1_robust_r471_result.json"
N = 32
KGRID = list(range(3, N, 2))          # 15 odd k
AGRID = [0.10, 0.05, 0.02, 0.01]
DELTA_CAL = 0.05
J_FAM = len(KGRID) * len(AGRID) + len(AGRID)   # 64
D_TEST = DELTA_CAL / J_FAM
SEED = 20260815


def hyper_pmf(K, k, x):
    if x < max(0, k - (N - K)) or x > min(k, K):
        return 0.0
    return comb(K, x) * comb(N - K, k - x) / comb(N, k)


def side(cnt, k):
    return 1 if cnt > k / 2 else 0


def build_flip_fixed():
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


def hoef_ucb(vals, delta):
    m = len(vals)
    return sum(vals) / m + sqrt(log(1 / delta) / (2 * m))


def mean_ci(vals, delta):
    m = len(vals)
    mu = sum(vals) / m
    r = sqrt(log(2 / delta) / (2 * m))
    return mu, r


def load_Ks(path):
    t = pq.read_table(path, columns=["problem", "pass_rate_72b_tir"]).to_pandas()
    t = t[t.pass_rate_72b_tir.notna() & (t.pass_rate_72b_tir != "n/a")].copy()
    t = t.drop_duplicates(subset=["problem"])
    Ks = [int(round(p * N)) for p in t.pass_rate_72b_tir.astype(float).tolist()]
    return Ks


def fit_prior(Ks, fit_idx):
    H = [0.0] * (N + 1)
    for i in fit_idx:
        H[Ks[i]] += 1.0
    return [h / len(fit_idx) for h in H]


def run_chain(Ks, H_prior, tag, cal_idx, test_idx):
    """Full CAL/TEST given a FROZEN prior H_prior and pre-fixed split indices."""
    n = len(Ks)

    cert_tab = build_cert_table(H_prior)
    f_fixed = build_flip_fixed()
    gad = {a: {K: dp_adaptive_flip(K, cert_tab, a) for K in range(N + 1)}
           for a in AGRID}

    # CAL selection
    cal = {}
    for k in KGRID:
        cal[("F", k)] = [f_fixed[Ks[i]][k] for i in cal_idx]
    for a in AGRID:
        cal[("B", a)] = [gad[a][Ks[i]][0] for i in cal_idx]

    sel = {}
    for a in AGRID:
        row = {}
        best = None
        for k in KGRID:
            if eb_ucb(cal[("F", k)], D_TEST) <= a:
                best = k
                break
        row["FIXED_EB_k"] = best
        row["FIXED_EB_cert"] = (round(eb_ucb(cal[("F", best)], D_TEST), 5) if best else None)
        bestH = None
        for k in KGRID:
            if hoef_ucb(cal[("F", k)], D_TEST) <= a:
                bestH = k
                break
        row["FIXED_HOEF_k"] = bestH
        row["FIXED_HOEF_cert"] = (round(hoef_ucb(cal[("F", bestH)], D_TEST), 5) if bestH else None)
        ub = eb_ucb(cal[("B", a)], D_TEST)
        row["BAYESH_cert"] = round(ub, 5)
        row["BAYESH_ok"] = bool(ub <= a)
        sel[str(a)] = row

    # TEST readout
    rules = []
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
    d_rule = 0.05 / n_rules

    test_res = {}
    bayesh_k_by_a = {}
    for name, kind, par, a in rules:
        flips, kss = [], []
        for i in test_idx:
            K = Ks[i]
            if kind == "F":
                flips.append(f_fixed[K][par]); kss.append(float(par))
            elif kind == "B":
                fl, ek = gad[par][K]; flips.append(fl); kss.append(ek)
            else:
                flips.append(0.0); kss.append(float(N))
        mu_f, r_f = mean_ci(flips, d_rule)
        mu_k, r_k = mean_ci(kss, d_rule)
        test_res[name] = {
            "alpha": a, "realized_flip": round(mu_f, 5), "flip_ci_radius": round(r_f, 5),
            "mean_k": round(mu_k, 3), "saving_vs_full": round(1 - mu_k / N, 4),
        }
        if kind == "B":
            bayesh_k_by_a[par] = kss

    gaps = {}
    for a in AGRID:
        r = sel[str(a)]
        if not (r["BAYESH_ok"] and r["FIXED_HOEF_k"]):
            continue
        kb = bayesh_k_by_a[a]
        diffs = [r["FIXED_HOEF_k"] - b for b in kb]
        mu_d, r_d = mean_ci([d / N for d in diffs], d_rule)
        gaps[str(a)] = {
            "bayesh_saving": test_res[f"BAYESH_a{a}"]["saving_vs_full"],
            "hoef_saving": test_res[f"FIXED_HOEF_a{a}"]["saving_vs_full"],
            "abs_gap_bayesh_minus_hoef": round(mu_d, 4),
            "gap_ci_radius": round(r_d, 4),
            "significant": bool(mu_d - r_d > 0),
        }

    # mixture structure summary
    mp = sum(Ks) / (n * N)
    ps = [K / N for K in Ks]
    var_p = sum((p - mp) ** 2 for p in ps) / n
    binom_base = sum(p * (1 - p) / N for p in ps) / n
    frac_ext = sum(1 for p in ps if p <= 0.1 or p >= 0.9) / n

    return {
        "tag": tag, "n": n, "n_cal": len(cal_idx), "n_test": len(test_idx),
        "p_mean": round(mp, 4), "p_var": round(var_p, 5),
        "binom_baseline_var": round(binom_base, 6),
        "het_excess_var": round(var_p - binom_base, 5),
        "frac_extreme": round(frac_ext, 4),
        "cal_selection": sel, "test_readout": test_res,
        "fair_gap_bayesh_vs_hoeffding": gaps,
        "n_rules": n_rules,
    }


def main():
    Ks1 = load_Ks(SHARD1)
    n1 = len(Ks1)
    rnd = random.Random(SEED)
    idx = list(range(n1))
    rnd.shuffle(idx)
    # shard1 split: FIT 4000 / CAL 4000 / TEST rest (disjoint)
    fit_idx_s1, cal_idx_s1, test_idx_s1 = idx[:4000], idx[4000:8000], idx[8000:]

    # R1: shard1 independent (prior from shard1 FIT, CAL/TEST on shard1)
    H_s1 = fit_prior(Ks1, fit_idx_s1)
    r1 = run_chain(Ks1, H_s1, "shard1_within", cal_idx_s1, test_idx_s1)

    # R2: transfer — prior from shard0 FIT (disjoint problems), CAL/TEST on shard1
    Ks0 = load_Ks(SHARD0)
    n0 = len(Ks0)
    rnd0 = random.Random(SEED)
    idx0 = list(range(n0))
    rnd0.shuffle(idx0)
    H_s0 = fit_prior(Ks0, idx0[:4000])
    r2 = run_chain(Ks1, H_s0, "shard1_transfer_prior_shard0", cal_idx_s1, test_idx_s1)

    out = {"seed": SEED, "N": N, "delta_cal": DELTA_CAL, "J_family": J_FAM,
           "R1_within": r1, "R2_transfer": r2}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    # compact readback
    for tag, r in (("R1_within", r1), ("R2_transfer", r2)):
        print(f"=== {tag} ===  n={r['n']} p_mean={r['p_mean']} het_excess={r['het_excess_var']} frac_extreme={r['frac_extreme']}")
        for a in AGRID:
            s = r["cal_selection"][str(a)]
            t = r["test_readout"].get(f"BAYESH_a{a}", {})
            g = r["fair_gap_bayesh_vs_hoeffding"].get(str(a), {})
            print(f"  a={a}: EBk={s['FIXED_EB_k']} HOEFk={s['FIXED_HOEF_k']} BHcert={s['BAYESH_cert']} BHok={s['BAYESH_ok']}"
                  f" | TEST flip={t.get('realized_flip')} k={t.get('mean_k')} save={t.get('saving_vs_full')}"
                  f" | gapBH-H={g.get('abs_gap_bayesh_minus_hoef')} sig={g.get('significant')}")


if __name__ == "__main__":
    main()
