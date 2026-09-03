#!/usr/bin/env python3
"""A11 r474: third cross-model carrier — RLVE Qwen3-4B-Thinking-2507 pass@8 (N=8).

Pre-registered questions (this docstring is the registration):
  (Q1) Between-structure fill: OMR (N=32, per-problem pass count) and OpenR1
       (M=2) closed the two extremes. Does the mixture-aware BAYES-H certificate
       land at an intermediate operating point on a real N=8 carrier with a
       THIRD model family (Qwen3-4B) and a different task family (RLVE
       verifiable-env counting/combinatorics, not competition math)?
  (Q2) Same-protocol readout: run the IDENTICAL FIT/CAL/TEST chain as r469/r471
       (J=64 family, delta=.05/64, EB/Hoeffding UCBs, exact hypergeometric DP)
       with N=8. Report CAL selection + single TEST readout at alpha in
       {.10,.05,.02,.01}. Hypothesis (from the mechanism claim): BAYES-H saving
       is intermediate between OpenR1 (32-50%) and OMR (73-81%), realized flip
       <= alpha everywhere the CAL certificate passes.
  (Q3) Descriptive drift diagnostic: sample_id is the generation order. Pool the
       trial->success slope across problems (linear regression on the 0/1
       sequence). Descriptive only; the main chain stays count-exchangeable
       replay (uniform random prefix without replacement), per the r469 MGR
       carrier-scope correction.

Carrier: CL-From-Nothing/RLVE-Qwen3-4B-Thinking-2507-Pass8-Rollouts
  HF snapshot eaeec946d8b5c61315f64335c830e9bddfe2eb46, license Apache-2.0.
  9000 questions x 8 samples; reward recomputed offline with the RLVE-Eval Gym
  verifier; reward in [-1,1], +1 = correct. Success := reward > 0 (a small mass
  of fractional rewards exists; they are verifier partial credit, counted as
  failure here — disclosed). 6 parquet shards pinned in SHA256SUMS.txt.

Readback: stdout + rlve_n8_r474_result.json. stdlib+pandas/pyarrow. No GPU/net.
"""
import json, math, random
from math import comb, log, sqrt
from collections import defaultdict, Counter
import pyarrow.parquet as pq

SHARDS = [f"rlve_p{i}.parquet" for i in range(6)]
OUT = "rlve_n8_r474_result.json"
N = 8
KGRID = list(range(3, N, 2))          # [3,5,7]
AGRID = [0.10, 0.05, 0.02, 0.01]
DELTA_CAL = 0.05
J_FAM = len(KGRID) * len(AGRID) + len(AGRID)   # 16
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
        f[K] = {k: sum(hyper_pmf(K, k, x) for x in range(k + 1)
                       if side(x, k) != full) for k in KGRID}
    return f


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


def load_data():
    """Return per-question K (#success of N=8), plus per-question ordered 0/1
    sequence for the descriptive drift diagnostic. Only questions with exactly
    N rows are kept."""
    succ = defaultdict(list)          # q -> [0/1 in sample_id order]
    envs = {}
    for sh in SHARDS:
        t = pq.read_table(sh, columns=["index", "sample_id", "rewards", "metadata"])
        d = t.to_pandas()
        for _, r in d.iterrows():
            succ[int(r["index"])].append((int(r["sample_id"]),
                                          1 if float(r["rewards"]) > 0 else 0))
            if int(r["index"]) not in envs:
                try:
                    envs[int(r["index"])] = json.loads(r["metadata"])["environment"]
                except Exception:
                    envs[int(r["index"])] = "?"
    Ks, seqs, qenv = [], [], []
    for q, pairs in succ.items():
        if len(pairs) != N:
            continue
        pairs.sort()
        Ks.append(sum(s for _, s in pairs))
        seqs.append([s for _, s in pairs])
        qenv.append(envs.get(q, "?"))
    return Ks, seqs, qenv


def fit_prior(Ks, fit_idx):
    H = [0.0] * (N + 1)
    for i in fit_idx:
        H[Ks[i]] += 1.0
    return [h / len(fit_idx) for h in H]


def drift_diagnostic(seqs):
    """Pooled trial->success slope across problems (descriptive)."""
    sx = sy = sxx = sxy = 0.0
    m = 0
    trials = list(range(N))
    tm = sum(trials) / N
    tv = sum((t - tm) ** 2 for t in trials)
    for s in seqs:
        sm = sum(s) / N
        for t, y in zip(trials, s):
            sxy += (t - tm) * (y - sm)
        sx += tv
        sy += sum((y - sm) ** 2 for y in s)
        m += 1
    slope = sxy / sx if sx else 0.0
    # per-position success rate
    pos = [sum(s[t] for s in seqs) / m for t in range(N)]
    return {"pooled_slope_per_trial": round(slope, 6),
            "per_position_success": [round(p, 4) for p in pos],
            "n_problems": m}


def run_chain(Ks, H_prior, tag, cal_idx, test_idx):
    n = len(Ks)
    cert_tab = build_cert_table(H_prior)
    f_fixed = build_flip_fixed()
    gad = {a: {K: dp_adaptive_flip(K, cert_tab, a) for K in range(N + 1)}
           for a in AGRID}

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

    rules = []
    for a in AGRID:
        r = sel[str(a)]
        if r["FIXED_EB_k"]:
            rules.append((f"FIXED_EB_a{a}", "F", r["FIXED_EB_k"], a))
        if r["FIXED_HOEF_k"]:
            rules.append((f"FIXED_HOEF_a{a}", "F", r["FIXED_HOEF_k"], a))
        if r["BAYESH_ok"]:
            rules.append((f"BAYESH_a{a}", "B", a, a))
    rules.append(("FULL8", "FULL", N, None))
    d_rule = 0.05 / len(rules)

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
        diffs = [(r["FIXED_HOEF_k"] - b) / N for b in kb]
        mu_d, r_d = mean_ci(diffs, d_rule)
        gaps[str(a)] = {
            "bayesh_saving": test_res[f"BAYESH_a{a}"]["saving_vs_full"],
            "hoef_saving": test_res[f"FIXED_HOEF_a{a}"]["saving_vs_full"],
            "abs_gap_bayesh_minus_hoef": round(mu_d, 4),
            "gap_ci_radius": round(r_d, 4),
            "significant": bool(mu_d - r_d > 0),
        }

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
        "K_dist": {str(k): sum(1 for K in Ks if K == k) for k in range(N + 1)},
        "cal_selection": sel, "test_readout": test_res,
        "fair_gap_bayesh_vs_hoeffding": gaps,
        "n_rules": len(rules),
    }


def main():
    Ks, seqs, qenv = load_data()
    n = len(Ks)
    print(f"loaded {n} full-8 questions; K dist:",
          sorted(Counter(Ks).items()))
    env_c = Counter(qenv)
    print("environments:", len(env_c), "top:", env_c.most_common(5))

    drift = drift_diagnostic(seqs)
    print("drift diag:", drift)

    rnd = random.Random(SEED)
    idx = list(range(n))
    rnd.shuffle(idx)
    fit_idx, cal_idx, test_idx = idx[:3000], idx[3000:6000], idx[6000:]
    print(f"split FIT={len(fit_idx)} CAL={len(cal_idx)} TEST={len(test_idx)}")

    H = fit_prior(Ks, fit_idx)
    print("prior H:", [round(h, 4) for h in H])
    res = run_chain(Ks, H, "rlve_within", cal_idx, test_idx)

    print(json.dumps({"cal_selection": res["cal_selection"]}, indent=1))
    for name, tr in res["test_readout"].items():
        print(f"TEST {name}: flip={tr['realized_flip']}±{tr['flip_ci_radius']} "
              f"mean_k={tr['mean_k']} saving={tr['saving_vs_full']}")
    print("fair gaps:", json.dumps(res["fair_gap_bayesh_vs_hoeffding"], indent=1))
    print(f"mixture: p_mean={res['p_mean']} het_excess={res['het_excess_var']} "
          f"frac_extreme={res['frac_extreme']}")

    out = {"seed": SEED, "N": N, "delta_cal": DELTA_CAL, "J_family": J_FAM,
           "carrier": "RLVE-Qwen3-4B-Thinking-2507-Pass8 (snapshot eaeec946, Apache-2.0)",
           "drift_diagnostic": drift,
           "env_counts": dict(env_c),
           "within": res}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
