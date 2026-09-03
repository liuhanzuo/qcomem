#!/usr/bin/env python3
"""A11 r483: extend the Proposition prop:tv TV-robustness certificate to the
OpenR1 carrier (M=2, DeepSeek-R1) -- the third and final carrier. Zero new
data, zero TEST re-reads: reuses the frozen FIT prior from the r473 pilot
(SEED=20260815, FIT=first 3000 of the 8853 usable problems; moment-matched
coarse 3-atom prior, SHA-256-pinned shard in earlystop_drift_r473).

Why this carrier is the analytic edge case of prop:tv: at M=2 the
identifiable support is exactly three atoms p in {0, .5, 1}, and
g(0)=g(1)=0 EXACTLY at every alpha (a decided prefix can never be
overturned by one more rollout), so the worst case over the TV ball is a
pure one-sided pump -- move mass into the unique positive-g atom p=.5:

    V(R) = Hmid*g_mid + g_mid*R        (R <= H0+H1 = 1-Hmid)

hence tau*(alpha) = alpha/g_mid - Hmid in closed form (or whole-simplex
when the rule never stops, g == 0).

Predictions registered in this docstring BEFORE running:
  P1: g-profile is alpha-dependent only through the stop decision at p=.5:
      g_mid = .25 (stop everywhere, alpha >= .07852... both certs <= a),
      g_mid = .125 (stop after pass-prefix only, .04598 <= a < .07852),
      g_mid = 0 (never stop, a < .04598). With certs .07852/.04598:
      alpha=.10 -> g_mid=.25; .05 -> .125; .02/.01 -> 0.
  P2: tau*(.10) = 4*.10 - .232 = 0.168 and tau*(.05) = 8*.05 - .232... no:
      tau* = alpha/g_mid - Hmid = .10/.25-.232 = .168; .05/.125-.232 = .168.
      The two levels coincide EXACTLY at 0.168 because g_mid halves with
      alpha across the rule change (g_mid = 2.5*alpha at both levels).
  P3: tau*(.02) = tau*(.01) = 1 (whole simplex): the never-stop rule has
      V(R) == 0 <= alpha for every prior. So tau* is NON-monotone in alpha
      across the rule-change boundary (0.168 -> 1.0 as alpha tightens from
      .05 to .02) -- the same phenomenon as RLVE's tau*(.05) > tau*(.10),
      starker: tightening alpha past the smallest certificate value makes
      the frozen rule trivially robust everywhere.
  P4: zero-g atom mass = H0+H1 = 76.8% at the stopping levels (and 100% at
      the never-stop levels) -- between OMR's 36-47% and RLVE's 74-87%,
      consistent with the r482 structural law (zero-g alignment drives
      robustness), here with an exact one-line V(R).

Cross-checks (self-audit):
  - closed form vs brute-force scan over the exact feasible split family
    (moved mass R split t0/t1 between the two zero-g atoms) -- 1e-9;
  - closed form vs scipy linprog (HiGHS) -- 1e-6 on a radius grid;
  - tau* by bisection-on-scan vs closed form -- 1e-3;
  - prior re-derived from the pinned parquet (same moment matching, seed,
    coin-stream order and split as r473) vs the r473 JSON-reported prior --
    5e-5 (the JSON is rounded to 4dp).

Carrier files: ../earlystop_drift_r473/all/default-00000-of-00010.parquet
(SHA-256 pinned in earlystop_drift_r473/SHA256SUMS.txt, HF snapshot
e4e141ec, MIT). Readback: stdout + tv_robustness_openr1_r483_result.json.
stdlib+pyarrow (+scipy only for the LP cross-check). No GPU/net.
"""
import json, os, random, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
R473 = os.path.join(HERE, "..", "earlystop_drift_r473")
OUT = os.path.join(HERE, "tv_robustness_openr1_r483_result.json")
SEED = 20260815
AGRID = [0.10, 0.05, 0.02, 0.01]
CERT_X0 = 0.07852  # frozen c_H(1,x=0) from r473 (posterior flip risk after a
CERT_X1 = 0.04598  # fail-prefix / pass-prefix); rules stop iff cert <= alpha


def g_profile(a):
    """Exact per-atom flip probability of the frozen BAYES-H rule at level a.
    At p in {0,1} the prefix is decided: stopping or not, no flip. At p=.5:
    flip w.p. .25 conditional on stopping after the observed prefix."""
    stop0, stop1 = CERT_X0 <= a, CERT_X1 <= a
    g_mid = 0.25 * (0.5 * stop0 + 0.5 * stop1)
    return [0.0, g_mid, 0.0], stop0, stop1


def worst_case_closed(H, g, R):
    """V(R) = base + g_mid * min(R, H0+H1): pump mass into the unique
    positive-g atom; both donor atoms have g=0 so the split is irrelevant.
    Exact for R <= H0+H1 = 1-Hmid (the q_mid <= 1 cap binds at the same
    point since the support sums to 1)."""
    base = sum(h * gv for h, gv in zip(H, g))
    gm = max(g)
    if gm <= 0:
        return base
    return base + gm * min(R, H[0] + H[2])


def worst_case_scan(H, g, R, steps=4001):
    """Brute force over the exact feasible family: move mass R into atom 1,
    split as t0 from atom 0 and t1 = R - t0 from atom 2 (the reverse
    direction -- pumping into a zero-g atom -- can only lower the value and
    is dominated; included implicitly since t in [0,R] covers all feasible
    forward pumps). Value is affine in t0, so the grid max is exact up to
    grid resolution g_mid * R / (steps-1)."""
    H0, Hm, H1 = H
    best = -1.0
    for i in range(steps):
        t0 = R * i / (steps - 1)
        t1 = R - t0
        if t0 > H0 + 1e-12 or t1 > H1 + 1e-12:
            continue
        q = (H0 - t0, Hm + R, H1 - t1)
        best = max(best, sum(a * b for a, b in zip(q, g)))
    return best


def lp_check(H, g, R):
    try:
        from scipy.optimize import linprog
        import numpy as np
    except Exception:
        return None
    n = len(H)
    c = -np.concatenate([np.array(g), np.zeros(2 * n)])
    A_eq = np.zeros((1 + n, 3 * n))
    A_eq[0, :n] = 1.0
    for K in range(n):
        A_eq[1 + K, K] = 1.0
        A_eq[1 + K, n + K] = -1.0
        A_eq[1 + K, 2 * n + K] = 1.0
    b_eq = np.concatenate([[1.0], np.array(H)])
    A_ub = np.zeros((1, 3 * n))
    A_ub[0, n:] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.array([2 * R]),
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return (-res.fun) if res.success else None


def critical_radius(H, g, alpha):
    """Bisection on the scan (grid-exact to ~1e-4), then the closed form."""
    base = sum(h * gv for h, gv in zip(H, g))
    if base > alpha:
        return 0.0, base, False, 0.0
    if max(g) <= alpha:
        return 1.0, base, True, 1.0
    lo, hi = 0.0, H[0] + H[2]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if worst_case_scan(H, g, mid, steps=2001) > alpha:
            hi = mid
        else:
            lo = mid
    gm = max(g)
    tau_cf = min((alpha - base) / gm, H[0] + H[2])
    return lo, base, False, tau_cf


def rederive_prior():
    """Re-fit the r473 coarse prior from the pinned parquet (same moment
    matching, seed and split) to confirm reproducibility of the frozen H."""
    import pyarrow.parquet as pq
    t = pq.read_table(os.path.join(R473, "all/default-00000-of-00010.parquet"),
                      columns=["problem", "correctness_math_verify",
                               "generations"]).to_pylist()
    probs = []
    for row in t:
        seen, obs = set(), []
        for gen, ok in zip(row["generations"], row["correctness_math_verify"]):
            gh = hashlib.sha1(gen.encode()).hexdigest()
            if gh in seen:
                continue
            seen.add(gh)
            obs.append(int(ok))
        if len(obs) == 2:
            probs.append(tuple(obs))
    rnd = random.Random(SEED)
    # r473 draws the per-problem fair-coin stream BEFORE the split shuffle;
    # reproduce that exact RNG order or the FIT split differs (self-audit:
    # omitting this line moves the prior by 0.012/0.016, caught by the
    # prior_rederived_match check).
    _coins = [rnd.random() for _ in range(len(probs))]
    idx = list(range(len(probs)))
    rnd.shuffle(idx)
    fit = idx[:3000]
    c0 = sum(1 for i in fit if sum(probs[i]) == 0) / len(fit)
    c1 = sum(1 for i in fit if sum(probs[i]) == 1) / len(fit)
    c2 = sum(1 for i in fit if sum(probs[i]) == 2) / len(fit)
    q = 2 * c1
    return [max(c0 - q / 4, 0.0), q, max(c2 - q / 4, 0.0)], len(probs)


def main():
    with open(os.path.join(R473, "openr1_m2_pilot_r473.json")) as f:
        r473 = json.load(f)
    H_json = [float(h) for h in r473["prior_H_p0_p5_p1"]]
    H_re, n_probs = rederive_prior()
    # r473 JSON stores the prior rounded to 4dp; compare at 5e-5 (rounding
    # tolerance), NOT 1e-9.
    prior_match = all(abs(a - b) < 5e-5 for a, b in zip(H_json, H_re))
    H = H_json
    H0, Hmid, H1 = H
    print(f"prior json={[round(h,4) for h in H_json]} "
          f"rederived={[round(h,4) for h in H_re]} match={prior_match} "
          f"n_problems={n_probs}")

    out = {"seed": SEED, "carrier": "OpenR1-Math-220k (M=2)", "N_atoms": 3,
           "alphas": AGRID,
           "definition": "V(R)=max over q in simplex, L1(q,Hhat)<=2R, of q.g; "
                         "one-sided-pump closed form, scan+LP cross-checked",
           "prior_H": [round(h, 6) for h in H],
           "prior_rederived_match": prior_match,
           "certs": {"x0": CERT_X0, "x1": CERT_X1}}

    gprof = {}
    for a in AGRID:
        g, s0, s1 = g_profile(a)
        gprof[str(a)] = {"stop_after_fail_prefix": s0,
                         "stop_after_pass_prefix": s1, "g": g,
                         "flip_hat": round(sum(h * gv for h, gv in zip(H, g)), 5)}
    out["g_profiles"] = gprof

    lp_grid = [0.005, 0.02, 0.042, 0.08, 0.12, 0.2]
    lpc, all_match = {}, True
    for a in AGRID:
        g = gprof[str(a)]["g"]
        rows = []
        for Rr in lp_grid:
            v_scan = worst_case_scan(H, g, Rr)
            v_cf = worst_case_closed(H, g, Rr)
            v_lp = lp_check(H, g, Rr)
            ok_cf = abs(v_cf - v_scan) < 1e-9
            ok_lp = (v_lp is not None and abs(v_lp - v_scan) < 1e-6)
            all_match = all_match and ok_cf and ok_lp
            rows.append({"R": Rr, "scan": round(v_scan, 9),
                         "closed_form": round(v_cf, 9),
                         "lp": (round(v_lp, 9) if v_lp is not None else None),
                         "scan_eq_cf": ok_cf, "scan_eq_lp": ok_lp})
        lpc[str(a)] = rows
    out["crosscheck"] = lpc
    out["crosscheck_all_match"] = all_match

    crit = {}
    for a in AGRID:
        g = gprof[str(a)]["g"]
        Rb, base, whole, Rcf = critical_radius(H, g, a)
        crit[str(a)] = {"tau_star_bisect": round(Rb, 5),
                        "tau_star_closed": round(Rcf, 5),
                        "agree_1e3": abs(Rb - Rcf) < 1e-3,
                        "flip_hat": round(base, 5),
                        "whole_simplex": bool(whole)}
    out["critical_radius"] = crit

    zg = {}
    for a in AGRID:
        g = gprof[str(a)]["g"]
        zeros = [K for K in range(3) if g[K] <= 1e-12]
        zg[str(a)] = {"H_mass_zero_g": round(sum(H[K] for K in zeros), 4),
                      "zero_g_atoms": zeros}
    out["zero_g_atoms"] = zg
    out["identifiability_note"] = (
        "at M=2 only pair-type rates are identifiable; het-excess variance "
        "is not defined on the coarse support -- the 3-atom moment-matched "
        "prior is the maximal identifiable heterogeneity")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("critical_radius:", json.dumps(crit, indent=1))
    print("zero_g:", json.dumps(zg, indent=1))
    print("g_profiles:", json.dumps({k: v["g"] for k, v in gprof.items()}))
    print("crosscheck_all_match:", all_match)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
