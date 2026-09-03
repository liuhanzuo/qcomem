"""r1897 corrected convex-minimax subgroup allocation (replaces r1895's flawed dual).

MGR instruction 593c907d2ccd: r1895's d* = lambda*R was wrong-dimensioned (lambda carries
R^{-3/2}); and minimax wrongly treated ALL candidates as the active set.  This runner:
  - builds the ENGINE coefficient matrix C[k,g] = w_g * beta_{j,g} over every engine
    k = (candidate j, deployed mixture w); beta_{j,g} = per-(candidate,group) FIT-only
    paired-diff std;
  - solves the joint convex-(n)-vs-concave-(mu) program via strong duality
    d* = R^{-1/2}[max_{mu in Delta_K} S(mu)]^{3/2}, S(mu) = sum_g a_g(mu)^{2/3},
    a_g(mu) = sum_k mu_k C[k,g];  active set = {k: mu*_k>0} by complementary slackness;
  - allocates n_g = R a_g(mu*)^{2/3}/S(mu*) (integer + floor + budget fill);
  - evaluates the SAME static paired-MPB certificate as M3 (delta=0.10, tau=0.04),
    comparable to uniform/neyman/width/sens, on the four frozen m3_cache carriers.

The old SUBGMIX_MINIMAX_R1895.json is preserved verbatim as INVALID-DIAGNOSTIC.

PROJECT A2_SAFE_MODEL_RANKING_SUBGROUP_MIX ROUND r1897  Pure CPU front / zero GPU.
"""
import json, os, sys, time
import numpy as np
from minimax_core_r1897 import minimax_solve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code'))
CACHE = os.path.join(ROOT, 'results', 'm3_cache')
OUT = os.path.join(ROOT, 'results', 'SUBGMIX_MINIMAX_R1897.json')

DELTA = 0.10
TAU = 0.04
FRACS = [0.50, 0.65, 0.80, 0.95]
SEEDS = [0, 1, 2]
CARRIERS = (sys.argv[1:] if len(sys.argv) > 1 else ['digits', 'fashion', 'mnist', 'news'])


def load_art(name, seed):
    z = np.load(os.path.join(CACHE, f'subgmmix_m3_{name}_s{seed}.npz'), allow_pickle=True)
    Mnames = [str(m) for m in z['Mnames']]
    errs = {m: z[f'e_{m}'].astype(bool) for m in Mnames}
    fit_err = {m: z[f'fe_{m}'].astype(bool) for m in Mnames}
    outer = {m: {int(g): float(v) for g, v in z[f'outer_{m}'].item().items()} for m in Mnames}
    yf = z['yf'].astype(int); yc = z['yc'].astype(int)
    return errs, fit_err, outer, yf, yc, Mnames


def beta_per_candidate(fit_err, yf, Mnames):
    """Per-(candidate j, group g) paired-diff std, FIT-only.

    beta[j,g] = mean over reference candidates i!=j of std_g(fit_err_i - fit_err_j).
    This is the width coefficient of engine (candidate j); drives UCB_{i*,j,g} half-width.
    """
    G = int(yf.max() + 1)
    fe = {m: fit_err[m].astype(float) for m in Mnames}
    M = len(Mnames)
    beta = {}
    for j in range(M):
        row = {}
        for g in range(G):
            sub = yf == g
            if sub.sum() < 2:
                row[g] = 0.0
                continue
            sds = []
            for a in range(M):
                if a == j:
                    continue
                sds.append((fe[Mnames[a]] - fe[Mnames[j]])[sub].std(ddof=1))
            row[g] = float(np.mean(sds)) if sds else 0.0
        beta[j] = row
    return beta, M, G


def grid_from_recs(name):
    r = np.load(os.path.join(CACHE, f'recs_{name}_s0.npz'), allow_pickle=True)
    return r['grid'].item()


def dcell(M, G):
    return DELTA / (M * (M - 1) * max(G, 1))


def build_ucb_mpb(errs, y_cal, rev, Mnames, G, dc):
    UCB = {}
    for i in Mnames:
        erri = errs[i].astype(float)
        for j in Mnames:
            if i == j:
                continue
            d = erri - errs[j].astype(float)
            for g in range(G):
                sub = d[(y_cal == g) & rev]
                n = int(sub.size)
                if n == 0:
                    UCB.setdefault((i, j), {})[g] = None
                    continue
                mu = sub.mean(); s = sub.std(ddof=1) if n > 1 else 0.0
                mX = (mu + 1.0) / 2.0; vX = (s ** 2) / 4.0; L = np.log(2.0 / dc)
                ubX = mX + np.sqrt(2.0 * vX * L / n) + 7.0 * L / (3.0 * (n - 1)) if n > 1 \
                    else mX + np.sqrt(L / 2.0)
                UCB.setdefault((i, j), {})[g] = 2.0 * ubX - 1.0
    return UCB


def select_cert(errs, y_cal, rev, UCB, Mnames, G, w):
    ptv = {}
    for i in Mnames:
        e = errs[i][rev]; yrev = y_cal[rev]
        pte = {g: float(e[yrev == g].mean()) if (yrev == g).sum() else 0.0 for g in range(G)}
        ptv[i] = sum(w[g] * pte[g] for g in range(G))
    i = min(ptv, key=ptv.get)
    ub = 0.0
    for j in Mnames:
        if j == i:
            continue
        b = sum(w[g] * UCB[(i, j)][g] for g in range(G))
        ub = max(ub, b)
    return i, ub


def engine_matrix(beta, grid, M, G):
    """C : (K=M*nW, G) engine coeff, engine k=(candidate j, mixture gp)."""
    rows = []
    labels = []
    for gp in grid:
        w = gp['w']
        for j in range(M):
            rows.append([w[g] * beta[j][g] for g in range(G)])
            labels.append((j, gp['name']))
    return np.array(rows, float), labels


def run_carrier(name, seed):
    errs, fit_err, outer, yf, yc, Mnames = load_art(name, seed)
    G = int(yc.max() + 1)
    grid = grid_from_recs(name)
    beta, M, G2 = beta_per_candidate(fit_err, yf, Mnames)
    assert G2 == G
    avail = {g: int((yc == g).sum()) for g in range(G)}
    order = {}
    for g in range(G):
        idx = np.where(yc == g)[0]
        rng = np.random.RandomState(1000 + seed * 10 + g)
        order[g] = rng.permutation(idx).tolist()
    C, _ = engine_matrix(beta, grid, M, G)
    rows_all = {}
    diag = {}
    for frac in FRACS:
        R = int(round(frac * len(yc)))
        R = min(R, sum(avail.values()))
        sol = minimax_solve(C, R, n_min=2, seed=seed)
        # floor each group to available
        n = {g: min(int(sol['n_g'][g]), avail[g]) for g in range(G)}
        # re-balance so sum == R (trim largest over-avail)
        over = sum(n.values()) - R
        while over > 0:
            gmax = max(range(G), key=lambda g: n[g] - min(n[g], 1))
            if n[gmax] <= 1:
                break
            n[gmax] -= 1; over -= 1
        while sum(n.values()) < R:
            gmin = min(range(G), key=lambda g: n[g])
            n[gmin] += 1
        rev = np.zeros(len(yc), bool)
        for g in range(G):
            if n[g] > 0:
                rev[order[g][:n[g]]] = True
        dc = dcell(len(Mnames), G)
        UCB = build_ucb_mpb(errs, yc, rev, Mnames, G, dc)
        rows = []
        for gp in grid:
            w = gp['w']
            i, ub = select_cert(errs, yc, rev, UCB, Mnames, G, w)
            trueR = {m: sum(w[g] * outer[m][g] for g in range(G)) for m in Mnames}
            reg = trueR[i] - min(trueR.values())
            rows.append({'w': gp['name'], 'chosen': i, 'UB': float(ub),
                         'true_regret': float(reg), 'label_cost': int(rev.sum()), 'n_g': n})
        rows_all[frac] = rows
        diag[frac] = {'active_set': sol['active_set'], 'n_active': len(sol['active_set']),
                      'K': M * len(grid), 'S_star': sol['S_star'],
                      'lambda': sol['lambda'], 'value_cont': sol['value_cont'],
                      'two_lambda_R': sol['two_lambda_R']}
    return G, grid, rows_all, beta, outer, diag


def main():
    t0 = time.time()
    ACC = {}
    DIAG = {}
    for name in CARRIERS:
        for seed in SEEDS:
            G, grid, rows_all, beta, outer, diag = run_carrier(name, seed)
            n_w = len(grid)
            DIAG[(name, seed)] = diag
            for frac, rows in rows_all.items():
                comm = [r for r in rows if r['UB'] <= TAU]
                cr = len(comm) / n_w if n_w else 0.0
                if comm:
                    regs = np.array([r['true_regret'] for r in comm])
                    cv = float(np.mean(regs <= TAU + 1e-9))
                else:
                    cv = float('nan')
                abst = [r for r in rows if r['UB'] > TAU]
                amean = float(np.mean([r['true_regret'] for r in abst])) if abst else None
                amax = float(np.max([r['true_regret'] for r in abst])) if abst else None
                ng = rows[0]['n_g']
                nva = np.array([ng[g] for g in range(G)])
                wid = np.median([r['UB'] for r in rows])
                ACC.setdefault((name, frac), []).append({
                    'seed': seed, 'carrier': name, 'frac': frac, 'n_w': n_w,
                    'committed_rate': cr, 'cert_validity': cv if cv == cv else None,
                    'abst_mean_regret': amean, 'abst_max_regret': amax,
                    'label_cost': float(rows[0]['label_cost']),
                    'n_g': nva.tolist(), 'med_UB': float(wid),
                    'beta_j0': [float(beta[0][g]) for g in range(G)]})
    agg = {}
    for (name, frac), vals in sorted(ACC.items()):
        crs = [v['committed_rate'] for v in vals]
        cvs = [v['cert_validity'] for v in vals if v['cert_validity'] is not None]
        agg[(name, frac)] = {
            'carrier': name, 'frac': frac, 'n_seeds': len(vals),
            'committed_rate': round(float(np.mean(crs)), 3),
            'cert_validity': round(float(np.mean(cvs)), 4) if cvs else None,
            'med_UB': round(float(np.mean([v['med_UB'] for v in vals])), 4),
            'label_cost': round(float(np.mean([v['label_cost'] for v in vals])), 1),
            'n_g_seed0': vals[0]['n_g'],
            'abst_mean_regret': round(float(np.mean([v['abst_mean_regret'] for v in vals if v['abst_mean_regret'] is not None])), 4) if any(v['abst_mean_regret'] is not None for v in vals) else None,
            'abst_max_regret': round(float(np.max([v['abst_max_regret'] for v in vals if v['abst_max_regret'] is not None], initial=-1e9)), 4) if any(v['abst_max_regret'] is not None for v in vals) else None,
        }
    diag_flat = {f'{k[0]}_s{k[1]}': v for k, v in DIAG.items()}
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1897',
           'tau': TAU, 'delta': DELTA, 'agg': [agg[k] for k in sorted(agg)],
           'diag_active_fracs': diag_flat,
           'theory_supersedes': 'THEORY_MINIMAX_R1895.md sec 2.3 (old d*=lambda R INVALID-DIAGNOSTIC; '
                                'corrected: value=2*lambda*R, value~R^{-1/2}, P*=d*=R^{-1/2}[max_u S(u)]^{3/2})',
           'code': 'subgmmix_minimax_r1897.py + minimax_core_r1897.py',
           'runtime_s': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('runtime_s', out['runtime_s'])
    for (name, frac), v in sorted(agg.items()):
        print(f"{name} frac={frac}: committed_rate={v['committed_rate']} "
              f"cv={v['cert_validity']} medUB={v['med_UB']} ng={v['n_g_seed0'][:6]}...")


if __name__ == '__main__':
    main()