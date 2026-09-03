#!/usr/bin/env python3
"""A11 r482: extend the Proposition prop:tv TV-robustness certificate to the
RLVE carrier (N=8, Qwen3-4B). Zero new data (same pinned parquet shards as
r474), zero TEST re-reads: certificate tables frozen from the same FIT prior
as r474 (SEED=20260815, shuffle idx, FIT=first 3000). Same math as
tv_robustness_r481.py, re-parameterized to the support size (r481 hardcodes
N=32 as a module global; here everything takes n = len(Hhat) - 1).

Pre-declared outputs (this docstring is the registration):
  - exact per-task flip g(K) and mean cost e_k(K) of the frozen BAYES-H rule
    at alpha in {.10,.05,.02,.01} (identical DP as r474);
  - V(R) closed form (two-sided greedy over the simplex intersect L1 ball),
    cross-checked against scipy linprog (HiGHS) on a grid of R;
  - critical radii tau*(alpha) = max{R : V(R) <= alpha};
  - zero-g atom mass (why V(R) grows slower than the Lipschitz rate);
  - context number: het_excess_var of this carrier (0.0999 at r474) vs OMR
    (0.129): the mechanism claim predicts SMALLER critical radii here because
    the middle-K mass is heavier (fewer zero-g atoms at moderate alpha? we
    measure, not assume).

Carrier files: ../earlystop_drift_r474/rlve_p{0..5}.parquet (SHA-256 pinned
in earlystop_drift_r474/SHA256SUMS.txt, HF snapshot eaeec946, Apache-2.0).

Readback: stdout + tv_robustness_rlve_r482_result.json.
stdlib+pandas/pyarrow (+scipy only for the LP cross-check). No GPU/net.
"""
import json, os, random, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
R474 = os.path.join(HERE, "..", "earlystop_drift_r474")
OUT = os.path.join(HERE, "tv_robustness_rlve_r482_result.json")
SEED = 20260815
AGRID = [0.10, 0.05, 0.02, 0.01]


def load_r474():
    spec = importlib.util.spec_from_file_location(
        "r474", os.path.join(R474, "rlve_n8_r474.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # main() is __name__-guarded; safe
    mod.SHARDS = [os.path.join(R474, f"rlve_p{i}.parquet") for i in range(6)]
    return mod


def worst_case(Hhat, g, R):
    """Exact LP optimum over simplex intersect L1 ball (radius 2R).
    Two-sided greedy; identical structure to r481, support-size agnostic."""
    n = len(Hhat) - 1
    base = sum(h * gv for h, gv in zip(Hhat, g))
    if R <= 0:
        return base

    def side(desc):
        order = sorted(range(n + 1), key=lambda K: -g[K] if desc else g[K])
        rem, acc = R, 0.0
        for K in order:
            room = (1.0 - Hhat[K]) if desc else Hhat[K]
            take = min(room, rem)
            if take <= 0:
                continue
            acc += take * g[K]
            rem -= take
            if rem <= 1e-15:
                break
        return acc, R - rem

    add, moved_a = side(True)
    take_, moved_t = side(False)
    if min(moved_a, moved_t) < R - 1e-12:
        return None
    return base + add - take_


def lp_check(Hhat, g, R):
    try:
        from scipy.optimize import linprog
        import numpy as np
    except Exception:
        return None
    n = len(Hhat)
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
    """tau*(alpha) = max{R : V(R) <= alpha} over R in [0,1] (the full simplex
    is reachable at R=1 since max L1 between two distributions is 2 = 2R).
    NOTE: r481 bisected on [0,0.5], which sufficed for OMR (tau* <= 0.157)
    but truncates RLVE, whose zero-g-heavy prior yields tau* up to ~0.55.
    If even the whole-simplex worst case max_K g(K) <= alpha, tau* = 1 and
    whole_simplex=True."""
    base = sum(h * gv for h, gv in zip(Hhat, g))
    if base > alpha:
        return 0.0, base, False
    if max(g) <= alpha:
        return 1.0, base, True
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        v = worst_case(Hhat, g, mid)
        if v is None or v > alpha:
            hi = mid
        else:
            lo = mid
    return lo, base, False


def main():
    r474 = load_r474()
    Ks, seqs, qenv = r474.load_data()
    n_all = len(Ks)
    rnd = random.Random(SEED)
    idx = list(range(n_all))
    rnd.shuffle(idx)
    fit_idx = idx[:3000]
    Hhat = r474.fit_prior(Ks, fit_idx)
    N = r474.N
    cert = r474.build_cert_table(Hhat)
    print(f"reloaded RLVE: {n_all} questions, FIT prior mass:",
          [round(h, 4) for h in Hhat])

    out = {"seed": SEED, "N": N, "alphas": AGRID, "carrier": "RLVE (N=8)",
           "definition": "V(R)=max over q in simplex, L1(q,Hhat)<=2R, of q.g; "
                         "two-sided greedy closed form, LP cross-checked",
           "prior_H": [round(h, 6) for h in Hhat]}
    tabs = {}
    for a in AGRID:
        g = [0.0] * (N + 1)
        ek = [0.0] * (N + 1)
        for K in range(N + 1):
            fl, km = r474.dp_adaptive_flip(K, cert, a)
            g[K] = fl
            ek[K] = km
        tabs[a] = {"g": g, "ek": ek,
                   "flip_hat": sum(Hhat[K] * g[K] for K in range(N + 1)),
                   "k_hat": sum(Hhat[K] * ek[K] for K in range(N + 1))}
    out["flip_hat"] = {str(a): round(tabs[a]["flip_hat"], 5) for a in AGRID}
    out["k_hat"] = {str(a): round(tabs[a]["k_hat"], 3) for a in AGRID}

    # zero-g atom mass per alpha
    zg = {}
    for a in AGRID:
        g = tabs[a]["g"]
        zeros = [K for K in range(N + 1) if g[K] <= 1e-12]
        zg[str(a)] = {"n_zero_g": len(zeros),
                      "H_mass_zero_g": round(sum(Hhat[K] for K in zeros), 4),
                      "zero_g_atoms": zeros}
    out["zero_g_atoms"] = zg

    # LP cross-check
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
    out["lp_crosscheck"] = lpc
    out["lp_crosscheck_all_match"] = all_match

    # worst case at the r471 transfer radius 0.042 and stress 0.10
    wc = {}
    for a in AGRID:
        g = tabs[a]["g"]
        wc[str(a)] = {f"V(R={R})": round(worst_case(Hhat, g, R), 5)
                      for R in (0.042, 0.10)}
        wc[str(a)]["valid_at_R0.042"] = bool(worst_case(Hhat, g, 0.042) <= a)
    out["worst_case_key_R"] = wc

    # critical radii (bisect on [0,1]; r481 used [0,0.5] which truncates RLVE)
    crit = {}
    for a in AGRID:
        Rstar, base, whole = critical_radius(Hhat, tabs[a]["g"], a)
        crit[str(a)] = {"tau_star": round(Rstar, 5),
                        "flip_hat": round(base, 5),
                        "whole_simplex": bool(whole)}
    out["critical_radius"] = crit

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("critical_radius:", json.dumps(crit, indent=1))
    print("worst_case_key_R:", json.dumps(wc, indent=1))
    print("zero_g:", json.dumps(zg, indent=1))
    print("lp_all_match:", all_match)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
