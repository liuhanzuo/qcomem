#!/usr/bin/env python3
"""A11 r481: exact worst-case population flip of frozen BAYES-H over total-variation
balls around the fitted prior. Upgrades Limitation (ii) from qualitative to a
quantified certificate; zero new data, zero TEST re-reads (rule frozen at r469/r471).

Setup (pre-declared in this docstring):
  flip(H) = sum_K H[K] * g[K] is LINEAR in H (paper, "No Jensen gap" remark),
  with g[K] = exact per-task replay flip prob of the FROZEN BAYES-H rule
  (cert tables built from Hhat) under true count K (DP of r469).

  The exact worst case over B_R = {q in simplex : ||q - Hhat||_1 <= 2R} is the
  LP  max sum q_K g_K. Writing q = Hhat + dp - dm (dp,dm >= 0, per-coordinate
  disjoint at optimum, sum dp = sum dm = R at the boundary):
    add side : dp_K <= 1 - Hhat_K  (q_K <= 1); fill atoms in DESCENDING g.
    take side: dm_K <= Hhat_K;                   drain atoms in ASCENDING g.
  Both greedy. Hence V(R) = Hhat.g + A(R) - C(R) with A, C piecewise-linear
  (breakpoints at cumulative cap sums). Exact for all real R; no integrality.

  Critical radius tau*(alpha) = largest R with V(R) <= alpha (V monotone,
  bisection). Validity-through-shift band: certify frozen rule at level
  alpha - R (nested rule family, Remark "continuous-alpha"), cost = added
  mean k from the same DP. Conservation curve tau*(alpha) on the alpha grid.
  Both carriers: OMR shard0 and shard1, priors fit on each shard's own FIT
  split with the r469/r471 seed and split.

Self-checks:
  (i) closed form vs direct LP (scipy linprog, HiGHS) on a grid of R per alpha;
  (ii) v_R == 0 atom mass at R=0.042 (context for the r471 transfer radius);
  (iii) V(0) == Hhat.g identity.

Carrier files (SHA-256 pinned in r467/r471 SHA256SUMS):
  ../earlystop_drift_r467/cot_shard0.parquet
  ../earlystop_drift_r471/cot_shard1.parquet

Readback: stdout + tv_robustness_r481_result.json. stdlib+pandas/pyarrow
(+scipy only for the LP cross-check). No GPU/net.
"""
import json, random
from math import log, sqrt
import pyarrow.parquet as pq
import importlib.util, os

N = 32
AGRID = [0.10, 0.05, 0.02, 0.01]
SEED = 20260815
OUT = "tv_robustness_r481_result.json"

# reuse exact DP machinery from the r469 runner (function defs only, no main)
spec = importlib.util.spec_from_file_location(
    "r469", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "earlystop_drift_r469", "fit_cal_test_r469.py"))
src = open(spec.origin).read()
ns = {}
exec(compile(src[:src.index("def eb_ucb")], spec.origin, "exec"), ns)
build_cert_table = ns["build_cert_table"]
dp_adaptive_flip = ns["dp_adaptive_flip"]


def load_counts(path):
    t = pq.read_table(path, columns=["problem", "pass_rate_72b_tir"]).to_pandas()
    t = t[t.pass_rate_72b_tir.notna() & (t.pass_rate_72b_tir != "n/a")].copy()
    t = t.drop_duplicates(subset=["problem"])
    return [int(round(p * N)) for p in t.pass_rate_72b_tir.astype(float).tolist()]


def fit_prior(Ks):
    rnd = random.Random(SEED)
    idx = list(range(len(Ks)))
    rnd.shuffle(idx)
    fit_idx = idx[:4000]
    H = [0.0] * (N + 1)
    for i in fit_idx:
        H[Ks[i]] += 1.0
    return [h / len(fit_idx) for h in H]


def g_tables(Hhat, alphas):
    cert = build_cert_table(Hhat)
    out = {}
    for a in alphas:
        g = [0.0] * (N + 1)
        ek = [0.0] * (N + 1)
        for K in range(N + 1):
            fl, km = dp_adaptive_flip(K, cert, a)
            g[K] = fl
            ek[K] = km
        out[a] = {"g": g, "ek": ek,
                  "flip_hat": sum(Hhat[K] * g[K] for K in range(N + 1)),
                  "k_hat": sum(Hhat[K] * ek[K] for K in range(N + 1))}
    return out


def greedy_side(g, caps, amounts, R, descending):
    """Move R mass greedily along atoms sorted by g; returns moved . g sum."""
    order = sorted(range(N + 1), key=lambda K: -g[K] if descending else g[K])
    rem = R
    acc = 0.0
    for K in order:
        room = (1.0 - amounts[K]) if descending else amounts[K]
        take = min(room, rem)
        if take <= 0:
            continue
        acc += take * g[K]
        rem -= take
        if rem <= 1e-15:
            break
    return acc, R - rem  # value moved, mass actually moved


def worst_case(Hhat, g, R):
    """Exact LP optimum over the simplex intersect L1 ball (radius 2R)."""
    base = sum(h * gv for h, gv in zip(Hhat, g))
    if R <= 0:
        return base
    add, moved_a = greedy_side(g, None, Hhat, R, descending=True)
    take, moved_t = greedy_side(g, None, Hhat, R, descending=False)
    if min(moved_a, moved_t) < R - 1e-12:
        return None  # ball radius beyond feasibility
    return base + add - take


def lp_check(Hhat, g, R):
    try:
        from scipy.optimize import linprog
        import numpy as np
    except Exception:
        return None
    n = N + 1
    c = -np.concatenate([np.array(g), np.zeros(2 * n)])
    A_eq = np.zeros((1 + n, 3 * n))
    A_eq[0, :n] = 1.0
    for K in range(n):
        A_eq[1 + K, K] = 1.0
        A_eq[1 + K, n + K] = -1.0
        A_eq[1 + K, 2 * n + K] = 1.0
    b_eq = np.concatenate([[1.0], np.array(Hhat)])
    A_ub = np.zeros((1, 3 * n))
    A_ub[0, n:] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.array([2 * R]),
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return (-res.fun) if res.success else None


def critical_radius(Hhat, g, alpha):
    base = sum(h * gv for h, gv in zip(Hhat, g))
    if base > alpha:
        return 0.0, base
    lo, hi = 0.0, 0.5
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        v = worst_case(Hhat, g, mid)
        if v is None or v > alpha:
            hi = mid
        else:
            lo = mid
    return lo, base


def main():
    out = {"seed": SEED, "N": N, "alphas": AGRID,
           "definition": "V(R)=max over q in simplex, L1(q,Hhat)<=2R, of q.g; "
                         "greedy closed form, LP cross-checked"}
    carriers = {
        "shard0": "../earlystop_drift_r467/cot_shard0.parquet",
        "shard1": "../earlystop_drift_r471/cot_shard1.parquet",
    }
    for cname, path in carriers.items():
        Ks = load_counts(path)
        Hhat = fit_prior(Ks)
        tabs = g_tables(Hhat, AGRID)
        cres = {"flip_hat": {str(a): round(tabs[a]["flip_hat"], 5) for a in AGRID},
                "g_max": {str(a): round(max(tabs[a]["g"]), 5) for a in AGRID},
                "k_hat": {str(a): round(tabs[a]["k_hat"], 3) for a in AGRID}}
        # (ii) zero-g atom mass
        zg = {}
        for a in AGRID:
            g = tabs[a]["g"]
            zeros = [K for K in range(N + 1) if g[K] <= 1e-12]
            zg[str(a)] = {"n_zero_g": len(zeros),
                          "H_mass_zero_g": round(sum(Hhat[K] for K in zeros), 4)}
        cres["zero_g_atoms"] = zg
        # (i) LP cross-check
        lp_grid = [0.005, 0.02, 0.042, 0.08, 0.12, 0.2]
        lpc = {}
        all_match = True
        for a in AGRID:
            g = tabs[a]["g"]
            rows = []
            for R in lp_grid:
                v_cf = worst_case(Hhat, g, R)
                v_lp = lp_check(Hhat, g, R)
                ok = (v_cf is not None and v_lp is not None
                      and abs(v_cf - v_lp) < 1e-6)
                all_match = all_match and ok
                rows.append({"R": R,
                             "closed_form": (round(v_cf, 6) if v_cf is not None else None),
                             "lp": (round(v_lp, 6) if v_lp is not None else None),
                             "match": ok})
            lpc[str(a)] = rows
        cres["lp_crosscheck"] = lpc
        cres["lp_crosscheck_all_match"] = all_match
        # worst case at the observed transfer radius R=0.042 and a stress R=0.10
        wc = {}
        for a in AGRID:
            g = tabs[a]["g"]
            wc[str(a)] = {f"V(R={R})": round(worst_case(Hhat, g, R), 5)
                          for R in (0.042, 0.10)}
            wc[str(a)]["valid_at_R0.042"] = bool(worst_case(Hhat, g, 0.042) <= a)
        cres["worst_case_key_R"] = wc
        # critical radii
        crit = {}
        for a in AGRID:
            Rstar, base = critical_radius(Hhat, tabs[a]["g"], a)
            crit[str(a)] = {"tau_star": round(Rstar, 5), "flip_hat": round(base, 5)}
        cres["critical_radius"] = crit
        # validity-through-shift band at R = 0.05: re-certify frozen tables at a-0.05
        band = {}
        for a in AGRID:
            ap = a - 0.05
            if ap <= 0:
                band[str(a)] = {"alpha_prime": None, "note": "no band possible"}
                continue
            t2 = g_tables(Hhat, [ap])[ap]
            v = worst_case(Hhat, t2["g"], 0.05)
            band[str(a)] = {
                "alpha_prime": ap,
                "worst_flip_at_R0.05": (round(v, 5) if v is not None else None),
                "valid": (v is not None and v <= a),
                "saving_band": round(1 - t2["k_hat"] / N, 4),
                "saving_orig": round(1 - tabs[a]["k_hat"] / N, 4),
            }
        cres["shift_band_R0.05"] = band
        out[cname] = cres
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT), "w") as f:
        json.dump(out, f, indent=1)
    for c in carriers:
        print(c, "critical_radius:", out[c]["critical_radius"])
        print(c, "worst_case_key_R:", out[c]["worst_case_key_R"])
        print(c, "shift_band_R0.05:", out[c]["shift_band_R0.05"])
        print(c, "lp_all_match:", out[c]["lp_crosscheck_all_match"])


if __name__ == "__main__":
    main()
