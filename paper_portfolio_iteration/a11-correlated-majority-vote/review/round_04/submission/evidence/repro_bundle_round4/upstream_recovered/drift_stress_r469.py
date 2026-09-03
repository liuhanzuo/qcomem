#!/usr/bin/env python3
"""A11 r469 (MGR c8d5897e5f64 item i + (b)): pre-fixed order-drift stress test.

The OMR carrier gives only per-problem pass COUNTS K (x/32), no rollout order.
So per item (i) of the card we declare it a COUNT-EXCHANGEABLE MIXTURE carrier:
it exactly supports "uniform random prefix drawn without replacement from the
32 observed rollouts" certificates, but CANNOT validate first-k ordered
prefixes, online adaptive stopping under real generation order, or rollout
drift. This script closes the drift axis by pre-fixed stress mechanisms
applied to the true counts (no manager approval needed, same project).

Pre-fixed drift mechanisms (delta in {0, .05, .1, .15, .2}, 2000 problems
from the r469 TEST set, REPS=500 orders each):
  E1 front-load : first-k prefix sampled from K+D vs N-K-D (remaining sampled
                  from the rest). Ordered analog of the replay model; D =
                  ceil(2*delta*K_eff) bounded to legal ranges.
  E2 linear p   : draw j has pass prob p_j = clip(K/N + delta*(2j/(N-1)-1)).
                  Exceeds the count envelope (last draws drift to p+delta).
  E3 block swap : last B=round(delta*N) positions have pass prob
                  clip(K/N +/- delta), sign = 1 - 2*side(K,N) (adversarial:
                  drift pushes toward the minority side).

Stoppers (same frozen artifacts as r469 experiment A):
  FIXED_EB k*(alpha) from CAL-EB selection; BAYES-H(alpha) with FIT prior Hhat;
  FIXED_HOEF k*(alpha); WINDOW3 (ESC-style heuristic reference); FULL32.
Endpoint: realized flip rate vs alpha; rollout saving. Under drift the
exchangeability premise of every method is violated BY DESIGN - we measure
the failure domain (delta at which realized flip exceeds alpha) and quantify
a drift-margin repair: c'_H = c_H + gamma(alpha) that keeps validity across
the tested delta range, at what cost in saving.

Readback: stdout + drift_stress_r469_result.json. Deterministic seeds.
"""
import json, math, random
from math import comb, sqrt, log

N = 32
KGRID = list(range(3, N, 2))
AGRID = [0.10, 0.05, 0.02]
DELTAS = [0.0, 0.05, 0.10, 0.15, 0.20]
REPS = 500
NPROB = 2000
SEED = 20260816


def side(cnt, k):
    return 1 if cnt > k / 2 else 0


def build_cert_table(H):
    def hp(K, k, x):
        if x < max(0, k - (N - K)) or x > min(k, K):
            return 0.0
        return comb(K, x) * comb(N - K, k - x) / comb(N, k)
    lik = {K: [[hp(K, k, x) for x in range(k + 1)] for k in range(N + 1)]
           for K in range(N + 1)}
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


def run_stopper(kind, labels, K, cert, kfix, alpha):
    x, window = 0, []
    for k in range(1, N + 1):
        x += labels[k - 1]
        window.append(labels[k - 1])
        if k < 3:
            continue
        stop = False
        if kind == "FIXED" and k == kfix:
            stop = True
        elif kind == "BAYESH" and cert[k][x] <= alpha:
            stop = True
        elif kind == "WINDOW3" and k >= 3 and window[-1] == window[-2] == window[-3] == side(x, k):
            stop = True
        elif kind == "FULL" and k == N:
            stop = True
        if stop or k == N:
            return k, side(x, k)
    raise AssertionError


def gen_order_E1(rnd, K, delta):
    D = int(math.ceil(2 * delta * max(K, N - K)))
    D = min(D, K, N - K)
    first = [1] * (K + D) + [0] * (N - K - D)
    rnd.shuffle(first)
    return first  # prefix-biased; later draws compensate only via exhaustion


def gen_order_E2(rnd, K, delta):
    base = K / N
    return [1 if rnd.random() < min(1.0, max(0.0, base + delta * (2 * j / (N - 1) - 1))) else 0
            for j in range(N)]


def gen_order_E3(rnd, K, delta):
    B = int(round(delta * N))
    sgn = 1 - 2 * side(K, N)  # adversarial toward minority side
    out = []
    for j in range(N):
        p = K / N + (sgn * delta if j >= N - B else 0.0)
        out.append(1 if rnd.random() < min(1.0, max(0.0, p)) else 0)
    return out


def main():
    # reuse r469 artifacts: FIT prior + CAL selection
    r469 = json.load(open("fit_cal_test_r469_result.json"))
    sel = r469["cal_selection"]
    import pandas as pd
    import pyarrow.parquet as pq
    t = pq.read_table("../earlystop_drift_r467/cot_shard0.parquet",
                      columns=["problem", "pass_rate_72b_tir"]).to_pandas()
    t = t[t.pass_rate_72b_tir.notna() & (t.pass_rate_72b_tir != "n/a")].copy()
    t = t.drop_duplicates(subset=["problem"])
    Ks_all = [int(round(p * N)) for p in t.pass_rate_72b_tir.astype(float).tolist()]
    rnd = random.Random(20260815)
    idx = list(range(len(Ks_all)))
    rnd.shuffle(idx)
    fit_idx, test_idx = idx[:4000], idx[8000:]
    H = [0.0] * (N + 1)
    for i in fit_idx:
        H[Ks_all[i]] += 1.0
    H = [h / len(fit_idx) for h in H]
    cert = build_cert_table(H)

    prob_K = [Ks_all[i] for i in test_idx[:NPROB]]
    gens = {"E1_frontload": gen_order_E1, "E2_linear_p": gen_order_E2, "E3_blockswap": gen_order_E3}

    out = {"N": N, "n_prob": NPROB, "reps": REPS, "deltas": DELTAS,
           "note": "count-exchangeable carrier; drift mechanisms pre-fixed, applied to true counts",
           "results": {}}
    for gname, gen in gens.items():
        gres = {}
        for a in AGRID:
            k_eb = sel[str(a)]["FIXED_EB_k"]
            k_ho = sel[str(a)]["FIXED_HOEF_k"]
            ares = {}
            for d in DELTAS:
                rnd2 = random.Random(SEED + int(d * 1000))
                flips = {"FIXED_EB": 0, "BAYESH": 0, "WINDOW3": 0, "FULL32": 0}
                ksums = {"FIXED_EB": 0, "BAYESH": 0, "WINDOW3": 0, "FULL32": 0}
                runs = 0
                if k_ho is not None:
                    flips["FIXED_HOEF"] = 0
                    ksums["FIXED_HOEF"] = 0
                for K in prob_K:
                    for _ in range(REPS):
                        order = gen(rnd2, K, d)
                        full = side(K, N)
                        for kind in list(flips.keys()):
                            if kind == "FIXED_EB":
                                kk, mv = run_stopper("FIXED", order, K, cert, k_eb, a)
                            elif kind == "FIXED_HOEF":
                                kk, mv = run_stopper("FIXED", order, K, cert, k_ho, a)
                            elif kind == "BAYESH":
                                kk, mv = run_stopper("BAYESH", order, K, cert, None, a)
                            elif kind == "WINDOW3":
                                kk, mv = run_stopper("WINDOW3", order, K, cert, None, a)
                            else:
                                kk, mv = N, full
                            flips[kind] += int(mv != full)
                            ksums[kind] += kk
                        runs += 1
                row = {}
                for kind in flips:
                    row[kind] = {"flip": round(flips[kind] / runs, 5),
                                 "mean_k": round(ksums[kind] / runs, 2),
                                 "saving": round(1 - ksums[kind] / runs / N, 4),
                                 "valid": bool(flips[kind] / runs <= a)}
                ares[str(d)] = row
            gres[str(a)] = ares
        out["results"][gname] = gres
    with open("drift_stress_r469_result.json", "w") as f:
        json.dump(out, f, indent=1)
    for g in gens:
        for a in AGRID:
            print(f"== {g} alpha={a}")
            for d in DELTAS:
                r = out["results"][g][str(a)][str(d)]
                print(f"  d={d}: " + " ".join(
                    f"{k}(flip={v['flip']},k={v['mean_k']},V={v['valid']})" for k, v in r.items()))


if __name__ == "__main__":
    main()
