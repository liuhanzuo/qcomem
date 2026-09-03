"""r1896: DECISIVE conditional-gate probe — single highest-variance vertex W'={e_g0}.

Executes the M3.5 conditional minimax gate on the four real carriers from the frozen
m3_cache. The theory (THEORY_MINIMAX_R1895.md  Sec.5 / Prop uniform-opt + the synthetic
counterexample) predicts: on a NON-SPANNING deployed set, convex-minimax should strictly
out-commit uniform, and the smallest such set is the SINGLE pure vertex on the
highest-variance subgroup e_g0  (g0 = argmax_g beta_g, beta only from FIT).

Why single vertex is the decisive test: with fewer vertices the difference between
"spread evenly" (uniform) and "concentrate on the queried subgroup" (minimax water-fill)
is maximized, and the capacity-wall is bounded from below for both.  The companion
top-3-vertex probe (SUBGMIX_CONDGATE_R1896.json) instead returns near-ties because
uniform already spreads labels onto those three groups too.

Soundness identical to M3.5: both allocations read FIT-only beta, reveal counts are
FIXED (no CAL stopping time), static paired-MPB certificate valid.  Same certificate,
budget, seed, delta=0.10, tau=0.04.

PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX  ROUND r1896  Pure CPU front / zero GPU.
"""
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code'))
from subgmmix_minimax_r1895 import (load_art, beta_from_fit, allocate_robustmm,
                                    build_ucb_mpb, select_cert, DELTA, TAU,
                                    SEEDS, CARRIERS, FRACS)

OUT = os.path.join(ROOT, 'results', 'SUBGMIX_CONDGATE_SINGLE_R1896.json')


def run_single(name, seed, frac):
    errs, fit_err, outer, yf, yc, Mnames = load_art(name, seed)
    G = int(yc.max() + 1)
    beta = beta_from_fit(fit_err, yf, Mnames)
    b = np.array([beta[g] for g in range(G)])
    g0 = int(np.argmax(b))
    # canonical CV: population CV over NONZERO-beta groups (matches r1895 matrix digits .42
    # / fashion .33 / mnist .24 / news .16); zeros (no paired variance) excluded so that
    # sparse-high-variance carriers like digits are not inflated by 0 entries.
    nz = b > 1e-6
    cv_beta = float(b[nz].std() / b[nz].mean()) if nz.mean() > 0 and abs(b[nz].mean()) > 1e-12 else 0.0
    Wp = [{'name': 'pure_top', 'w': {g: (1.0 if g == g0 else 0.0) for g in range(G)}}]
    avail = {g: int((yc == g).sum()) for g in range(G)}
    R = min(int(round(frac * len(yc))), sum(avail.values()))
    dc = DELTA / (len(Mnames) * (len(Mnames) - 1) * max(G, 1))
    order_g = {g: np.random.RandomState(1000 + seed * 10 + g).permutation(
        np.where(yc == g)[0]).tolist() for g in range(G)}
    rows = {}
    for rule in ('uniform', 'minimax'):
        if rule == 'uniform':
            n = {g: max(1, int(round(R / G))) for g in range(G)}
            rem = R - sum(n.values()); gs = list(range(G))
            for k in range(rem):
                n[gs[k]] += 1
        else:
            n, wstar = allocate_robustmm(beta, avail, R, Wp)
        rev = np.zeros(len(yc), bool)
        for g in range(G):
            for k in range(min(n[g], len(order_g[g]))):
                rev[order_g[g][k]] = True
        UCB = build_ucb_mpb(errs, yc, rev, Mnames, G, dc)
        i, ub = select_cert(errs, yc, rev, UCB, Mnames, G, Wp[0]['w'])
        trueR = float(outer[i][g0])
        minR = min(float(outer[m][g0]) for m in Mnames)
        rows[rule] = {'g0': g0, 'n_g0': int(n[g0]), 'R': int(R),
                      'chosen': i, 'UB': float(ub), 'true_regret_g0': float(trueR - minR),
                      'committed': bool(ub <= TAU), 'top_outer': trueR, 'min_outer': minR,
                      'cv_beta': cv_beta}
    return g0, cv_beta, rows


def main():
    t0 = time.time()
    ACC = {}
    for name in CARRIERS:
        for seed in SEEDS:
            for frac in FRACS:
                g0, cv, rows = run_single(name, seed, frac)
                for rule, r in rows.items():
                    ACC.setdefault((name, frac, rule), []).append({
                        'seed': seed, 'committed': r['committed'],
                        'UB': r['UB'], 'true_regret_g0': r['true_regret_g0'],
                        'n_g0': r['n_g0'], 'g0': g0, 'cv_beta': cv})
    agg = {}
    for (name, frac, rule), vals in sorted(ACC.items()):
        cmts = [v['committed'] for v in vals]
        UBs = [v['UB'] for v in vals]
        regs = [v['true_regret_g0'] for v in vals]
        agg[(name, frac, rule)] = {
            'carrier': name, 'frac': frac, 'rule': rule, 'n_seeds': len(vals),
            'g0': vals[0]['g0'],
            'cv_beta': round(float(np.mean([v['cv_beta'] for v in vals])), 4),
            'n_g0_ratio': round(float(np.mean([v['n_g0'] / v['n_g0'] for v in vals])), 2),
            'commit_rate': round(float(np.mean(cmts)), 3),
            'med_UB': round(float(np.median(UBs)), 4),
            'mean_regret_when_committed': round(float(np.mean([v['true_regret_g0']
                for v in vals if v['committed']])), 4) if any(v['committed'] for v in vals) else None,
            'max_regret_when_committed': round(float(np.max([v['true_regret_g0']
                for v in vals if v['committed']])), 4) if any(v['committed'] for v in vals) else None,
        }
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1896',
           'kind': 'conditional_gate_single_highest_variance_vertex',
           'tau': TAU, 'delta': DELTA, 'fracs': FRACS,
           'agg': [agg[k] for k in sorted(agg)],
           'theory': 'THEORY_MINIMAX_R1895.md (Sec.5 conditional selection)',
           'runtime_s': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"{'carrier':8}{'frac':>6} {'uniform cmt/medUB':>20} {'minimax cmt/medUB':>22}")
    for name in CARRIERS:
        for frac in FRACS:
            u = agg[(name, frac, 'uniform')]; m = agg[(name, frac, 'minimax')]
            print(f"{name:8}{frac:>6} {u['commit_rate']:>7.3f}/{u['med_UB']:.4f}  "
                  f"{m['commit_rate']:>7.3f}/{m['med_UB']:.4f}")


if __name__ == '__main__':
    main()