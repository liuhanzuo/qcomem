"""Diagnostic: quantify the adaptive (data-dependent-stopping) certificate bandwidth price on M3.

r1886 M3 mnist authoritative run showed adaptive (CS track) committed=0 everywhere while
pre-specified uniform reached 100% committed.  This script asks WHY:
  (a) is the anytime-valid Hoeffding CS the bottleneck (too-wide bandwidth -> fixed by EB-CS)?
  (b) or does the adaptive allocation itself fail to buy a tau-certifiable certificate on
      low-error carriers?
We run the SAME adaptive trajectory (reuse cached mnist s0 matrix) and at the terminal reveal
counts evaluate three bandwidth families on the worst-case-mixture paired-regret UB:
  - Hoeffding anytime-CS (what the current runner uses; honest, subscription per-n delta split)
  - empirical-Bernstein anytime-CS (Bennett-style, variance-shrunk, delta split per n)
  - static MPB at terminal n (NOT honest for adaptive stopping; a diagnostic UPPER bound on
    what the tightest valid finite-sample certificate could be)
Status of certificate soundness is STRICTLY separated from committed coverage.

PROJECT A2_SAFE_MODEL_RANKING_SUBGROUP_MIX  ROUND r1886  Pure CPU / front / zero GPU.
"""
import json, numpy as np, sys, time
sys.path.insert(0, 'subgroup_mix_ranking/code')
from subgmmix_m3_budget_r1886 import (load_artifacts_placeholder, w_grid, dcell_cs,
                                      dcell_static, delta_scale_hoeff, _adaptive_sample)


def placeholders():
    return None


def hoeff_tw(dcell, n):
    dn = dcell * 6.0 / (np.pi ** 2 * n * (n + 1))
    return np.sqrt(2.0 * np.log(2.0 / dn) / n)


def eb_tw(dcell, n, v):
    """empirical-Bernstein anytime-style one-sided (Bennett), delta split per n for time-uniformity.
    d in [-1,1]; mean m=(mu+1)/2 in [0,1], variance <= v/4.  mu_ub = m + sqrt(2 v b/n) + 7 b/(3(n-1))."""
    b = np.log(2.0 / (dcell * 6.0 / (np.pi ** 2 * n * (n + 1))))
    m = (v_use + 1.0) / 2.0 if False else None
    # need actual variance; return lambda-friendly
    return b


def ub_hoeff(sub, dc, n):
    return sub.mean() + hoeff_tw(dc, n)


def ub_eb(sub, dc, n):
    v = sub.var(ddof=1) if n > 1 else 0.0
    b = np.log(2.0 / (dc * 6.0 / (np.pi ** 2 * n * (n + 1))))
    return sub.mean() + np.sqrt(2.0 * v * b / n) + 7.0 * b / (3.0 * (n - 1))


def ub_mpb(sub, dc, n):
    """static MPB at terminal n (honest ONLY for pre-specified; diagnostic here)."""
    mu = sub.mean(); m = (mu + 1.0) / 2.0; s = sub.std(ddof=1) if n > 1 else 0.0
    vX = (s ** 2) / 4.0; L = np.log(2.0 / dc)
    ubX = (m + np.sqrt(2.0 * vX * L / n) + 7.0 * L / (3.0 * (n - 1))) if n > 1 else (m + np.sqrt(L / 2))
    return 2.0 * ubX - 1.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--carrier', default='mnist')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frac', type=float, default=0.95)
    ap.add_argument('--tau', type=float, default=0.04)
    a = ap.parse_args()
    t0 = time.time()

    CARRIERS = [a.carrier]; SEEDS = [a.seed]; FRACS = [a.frac]; TAU = a.tau
    DELTA = 0.10
    # import artifacts via the m3 runner's cached loader by importing the cached npz directly
    import os
    # reuse the m3 runner module's get_artifacts (works off cache)
    sys.path.insert(0, 'subgroup_mix_ranking/code')
    from subgmmix_m3_budget_r1886 import run_carrier  # heavy; but cached carrier skips training
    print("(placeholder for full adaptive-CS-width comparison — see code.)")
    savepath = f"subgroup_mix_ranking/results/SUBGMIX_M3_ABW_DIAG_{a.carrier}_s{a.seed}.json"
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1886-abw',
           'carrier': a.carrier, 'seed': a.seed, 'frac': a.frac,
           'runtime_s': round(time.time() - t0, 1)}
    with open(savepath, 'w') as f:
        json.dump(out, f, indent=2)
    print('saved placeholder', savepath, 'runtime_s', out['runtime_s'])


if __name__ == '__main__':
    main()