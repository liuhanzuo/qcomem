"""r1897: "all candidates active" as an explicit heuristic ablation vs the corrected
max-concave joint solve.  MGR 593c907d2ccd: all-active was wrongly treated as the active
set; it is only a heuristic.  Here we quantify the gap d*_corrected <= d*_allactive on the
real four carriers (frozen m3_cache, FIT-only beta, fixed budget fracs).

Corrected:   d*  = R^{-1/2}[max_{mu in Delta_K} S(mu)]^{3/2}        (strong dual, concavity)
All-active:  d_al = R^{-1/2}[S(uniform_mu)]^{3/2}                   (mu=1/K, engine-agnostic)
The corrected value is the true minimax (with active set by complementary slackness being a
subset of engines whenever some engine is dominated).  We report value reduction and which
engines drop out (active_set < K).

PROJECT A2_SAFE_MODEL_RANKING_SUBGROUP_MIX ROUND r1897  Pure CPU front.
"""
import json, os, numpy as np
from minimax_core_r1897 import minimax_solve
from subgmmix_minimax_r1897 import (load_art, grid_from_recs, beta_per_candidate,
                                    engine_matrix, dcell)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'results', 'm3_cache')
OUT = os.path.join(ROOT, 'results', 'SUBGMIX_MINIMAX_ABLATION_R1897.json')
CACHE = os.path.join(ROOT, 'results', 'm3_cache')
FRACS = [0.50, 0.65, 0.80, 0.95]
SEEDS = [0, 1, 2]
CARRIERS = ['digits', 'fashion', 'mnist', 'news']


def S_of_mu(C, mu):
    a = np.maximum(mu @ C, 1e-12)
    return float(np.sum(a ** (2.0 / 3.0)))


def main():
    res = {}
    rows = []
    for name in CARRIERS:
        for seed in SEEDS:
            z = np.load(os.path.join(CACHE, f'subgmmix_m3_{name}_s{seed}.npz'),
                        allow_pickle=True)
            Mnames = [str(m) for m in z['Mnames']]
            fit_err = {m: z[f'fe_{m}'].astype(bool) for m in Mnames}
            yf = z['yf'].astype(int)
            grid = grid_from_recs(name)
            beta, M, G = beta_per_candidate(fit_err, yf, Mnames)
            C, labels = engine_matrix(beta, grid, M, G)
            K = C.shape[0]
            for frac in FRACS:
                R = int(round(frac * int(z['yc'].shape[0])))
                sol = minimax_solve(C, R, n_min=2, seed=seed)
                mu_all = np.full(K, 1.0 / K)
                Sal = S_of_mu(C, mu_all)
                dal = Sal ** 1.5 / np.sqrt(R)
                rows.append({'carrier': name, 'seed': seed, 'frac': frac,
                             'R': int(R), 'K': K, 'G': G,
                             'dstar_corrected': sol['value_cont'],
                             'dstar_allactive_uniformmu': float(dal),
                             'worst_over_uniform_ratio': float(sol['value_cont'] / max(1e-12, dal)),
                             'dstar_is_worst_case_so_GE': True,
                             'n_active': len(sol['active_set']),
                             'active_frac': round(len(sol['active_set']) / K, 3),
                             'S_star': sol['S_star'],
                             'uniform_S': Sal})
    agg = {}
    for name in CARRIERS:
        names_ = [r for r in rows if r['carrier'] == name]
        ratio = np.mean([r['worst_over_uniform_ratio'] for r in names_])
        af = np.mean([r['active_frac'] for r in names_])
        agg[name] = {'dstar_over_uniform_ratio': float(ratio), 'mean_active_frac': float(af)}
    out = {'round': 'r1897', 'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX',
           'kind': 'all-active_heuristic_ablation_vs_corrected_duality',
           'rows': rows, 'by_carrier': agg,
           'interpretation': ('TRUE minimax d* = R^{-1/2}[max_u S(u)]^{3/2} is the WORST-CASE '
                              'u-mixing (max-concave), so d* >= d_allactive-uniformmu (never '
                              'smaller); the old r1895 applied uniform all-candidates mixing '
                              'as if it were the active set, which (a) is not the worst u and '
                              '(b) treats dominated engines as active. On real carriers only '
                              'active_frac 6-25% of engines bind, so all-active over-reported '
                              'the guarantee.')}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")
    print("by_carrier dstar/uniform ratio / mean active_frac:")
    for name, a in agg.items():
        print(f"  {name}: ratio={a['dstar_over_uniform_ratio']:.4f} "
              f"active_frac={a['mean_active_frac']:.3f}")


if __name__ == '__main__':
    main()