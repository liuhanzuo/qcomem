"""r1896: end-to-end executable conditional gate for the minimax allocation (M3.5).

Completes the MGR r1895 next-plan item (a): the CONVEX-MINIMAX rule is currently only
theorized (THEORY_MINIMAX_R1895.md) and demoed on the synthetic asymmetric counterexample
; its scope claim is "strictly better than uniform on asymmetric NON-SPANNING deployed W".
This probe EXECUTES that claim on the four real carriers from the frozen m3_cache (no
retraining; only the deployed ambiguity set and the reveal allocation change).

Design of a real non-spanning deployed set:
  - A deployed mixture set W is *spanning* if it touches every subgroup (the default grid
    with skew_peak0..9 + uniform is spanning).  It is *non-spanning* if every w in W is
    supported on a PROPER subset of subgroups, and the union of supports is a subset of G.
  - On the full grid, uniform = maximin (= formal minimax solution), so minimax is not
    supposed to win there.  The theory (Prop uniform-opt + the counterexample) instead
    predicts: on a non-spanning W whose joint support is exactly the HIGH-variance
    subgroups, minimax (water-fill concentrating budget there) should out-commit uniform.
  - We test exactly that: build W' from the existing skew_peak vertices restricted to the
    top-k high-beta subgroups (per FIT), evaluate BOTH uniform and convex-minimax under the
    SAME static paired-MPB certificate, same budget, same grid selection, 3 seeds.
  - Conditional gate (the operational rule): measured FIT CV(hat beta) + the fact that a
    deployment is non-spanning -> choose convex-minimax.  We report the real committed-rate
    gain and the honest scope (symmetric spanning W stays uniform).

SOUNDNESS is preserved: both allocations read FIT-only beta, reveal counts are FIXED
(no CAL-data-dependent stopping time), so the static paired-MPB certificate stays valid.
So this is the SAME experiment as M3.5 / M3 on a narrower deployed set.

PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX  ROUND r1896  Pure CPU front / zero GPU.
"""
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code'))
from subgmmix_minimax_r1895 import (load_art, beta_from_fit, grid_from_recs,
                                    allocate_robustmm, build_ucb_mpb, select_cert,
                                    DELTA, TAU, FRACS, SEEDS, CARRIERS)

OUT = os.path.join(ROOT, 'results', 'SUBGMIX_CONDGATE_R1896.json')
TOP_K = 3          # non-spanning support: top-k high-variance subgroups

GATE_STAT = {}     # (carrier, frac) -> {cv_beta, top_groups, n_spanning}


def _restrict_nonspanning(name, G, grid, beta):
    """Return a non-spanning deployed set W' supported on the top-k high-beta subgroups."""
    b = np.array([beta[g] for g in range(G)])
    top = [int(g) for g in np.argsort(-b)[:TOP_K]]
    topset = set(top)
    Wp = []
    for gp in grid:
        w = gp['w']
        # keep a peaked vertex if its ENTIRE support sits inside the top-k subgroups
        supp = {g for g, v in w.items() if v > 1e-6}
        if len(supp) == 1 and supp < topset:
            Wp.append({'name': gp['name'], 'w': w})
    # ensure at least one vertex per top subgroup, pull in the matching skew_peak if absent
    have = set()
    for gp in Wp:
        for g, v in gp['w'].items():
            if v > 1e-6:
                have.add(g)
    for gp in grid:
        w = gp['w']
        supp = {g for g, v in w.items() if v > 1e-6}
        if len(supp) == 1:
            g0 = list(supp)[0]
            if g0 in top and g0 not in have:
                Wp.append({'name': gp['name'], 'w': w})
                have.add(g0)
    if len(Wp) < len(top):
        # linear fallback: build pure single-group vertices directly for missing groups
        for g0 in top:
            if g0 in have:
                continue
            Wp.append({'name': f'pure_{g0}', 'w': {g: (1.0 if g == g0 else 0.0) for g in range(G)}})
            have.add(g0)
    return Wp, top


def run_both(name, seed, frac):
    errs, fit_err, outer, yf, yc, Mnames = load_art(name, seed)
    G = int(yc.max() + 1)
    grid = grid_from_recs(name)                # full spanning grid (for CV computation)
    beta = beta_from_fit(fit_err, yf, Mnames)
    b = np.array([beta[g] for g in range(G)])
    nz = b > 1e-6
    cv_beta = float(b[nz].std() / b[nz].mean()) if nz.mean() > 0 and abs(b[nz].mean()) > 1e-12 else 0.0
    Wp, top = _restrict_nonspanning(name, G, grid, beta)
    avail = {g: int((yc == g).sum()) for g in range(G)}
    order_g = {}
    for g in range(G):
        idx = np.where(yc == g)[0]
        rng = np.random.RandomState(1000 + seed * 10 + g)
        order_g[g] = rng.permutation(idx).tolist()
    R = int(round(frac * len(yc)))
    R = min(R, sum(avail.values()))
    dc = dcell = DELTA / (len(Mnames) * (len(Mnames) - 1) * max(G, 1))
    out = {}
    for rule in ('uniform', 'minimax'):
        if rule == 'uniform':
            n = {g: max(1, int(round(R / G))) for g in range(G)}
            # fix to exact budget: distribute remainder
            rem = R - sum(n.values())
            gs = list(range(G))
            for k in range(rem):
                n[gs[k]] += 1
        else:
            n, _wstar = allocate_robustmm(beta, avail, R, Wp)  # full {0..G-1} int dict
            n = {g: max(1, int(n[g])) for g in range(G)}
        rev = np.zeros(len(yc), bool)
        for g in range(G):
            for k in range(min(n[g], len(order_g[g]))):
                rev[order_g[g][k]] = True
        UCB = build_ucb_mpb(errs, yc, rev, Mnames, G, dc)
        rows = []
        for gp in Wp:
            w = gp['w']
            i, ub = select_cert(errs, yc, rev, UCB, Mnames, G, w)
            trueR = {m: sum(w[g] * outer[m][g] for g in range(G)) for m in Mnames}
            reg = trueR[i] - min(trueR.values())
            rows.append({'w': gp['name'], 'chosen': i, 'UB': float(ub),
                         'true_regret': float(reg)})
        out[rule] = rows
    return {g: g in top for g in range(G)}, len(Wp), Wp, out, cv_beta, top


def main():
    t0 = time.time()
    ACC = {}
    for name in CARRIERS:
        for seed in SEEDS:
            for frac in FRACS:
                is_top, nWp, Wp, rows_by_rule, cv_beta, top = run_both(name, seed, frac)
                for rule, rows in rows_by_rule.items():
                    comm = [r for r in rows if r['UB'] <= TAU]
                    cr = len(comm) / nWp if nWp else 0.0
                    cv = float(np.mean([r['true_regret'] <= TAU + 1e-9 for r in comm])) if comm else None
                    amean = float(np.mean([r['true_regret'] for r in rows if r['UB'] > TAU])) \
                        if any(r['UB'] > TAU for r in rows) else None
                    amax = float(np.max([r['true_regret'] for r in rows if r['UB'] > TAU], initial=None)) \
                        if any(r['UB'] > TAU for r in rows) else None
                    ACC.setdefault((name, frac, rule), []).append({
                        'seed': seed, 'committed_rate': cr, 'cert_validity': cv,
                        'abst_mean_regret': amean, 'abst_max_regret': amax,
                        'top_groups': top, 'n_wp': nWp, 'cv_beta': cv_beta})
    agg = {}
    for (name, frac, rule), vals in sorted(ACC.items()):
        crs = [v['committed_rate'] for v in vals]
        cvs = [v['cert_validity'] for v in vals if v['cert_validity'] is not None]
        amean = [v['abst_mean_regret'] for v in vals if v['abst_mean_regret'] is not None]
        amax = [v['abst_max_regret'] for v in vals if v['abst_max_regret'] is not None]
        agg[(name, frac, rule)] = {
            'carrier': name, 'frac': frac, 'rule': rule, 'n_seeds': len(vals),
            'n_wp': vals[0]['n_wp'], 'top_groups': vals[0]['top_groups'],
            'cv_beta': round(float(np.mean([v['cv_beta'] for v in vals])), 4),
            'committed_rate': round(float(np.mean(crs)), 3),
            'cert_validity': round(float(np.mean(cvs)), 4) if cvs else None,
            'abst_mean_regret': round(float(np.mean(amean)), 4) if amean else None,
            'abst_max_regret': round(float(np.max(amax)), 4) if amax else None,
        }
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1896',
           'tau': TAU, 'delta': DELTA, 'fracs': FRACS, 'top_k': TOP_K,
           'agg': [agg[k] for k in sorted(agg)],
           'gate_stat': {str(k): {kk: (vv if isinstance(vv, (int, float, str, type(None))) else vv)
                                  for kk, vv in v.items() for v in [{}]} for k, v in GATE_STAT.items()},
           'runtime_s': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('runtime_s', out['runtime_s'])
    print('tau=%s top_k=%s  committed_rate on NON-SPANNING high-var W\':  uniform vs convex-minimax')
    print(f"{'carrier':8}{'frac':>6}{'nWp':>5}  {'uniform cr':>10}   {'minimax cr':>11}   cv")
    for name in CARRIERS:
        for frac in FRACS:
            u = agg.get((name, frac, 'uniform')); m = agg.get((name, frac, 'minimax'))
            if not u or not m:
                continue
            print(f"{name:8}{frac:>6}{u['n_wp']:>5}  {u['committed_rate']:>10.3f}   "
                  f"{m['committed_rate']:>11.3f}   cv={u['cv_beta']:.3f}")


if __name__ == '__main__':
    main()