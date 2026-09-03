"""r1895: convex-minimax subgroup allocation as a strong same-budget baseline.

MGR instruction 809f17cdbe78: formalize fixed-budget minimax
  min_{n_g, sum n_g = R} max_j sum_g w_g r_g(n_g)
(NB here the ambiguity set is over BOTH the deployed grid W and the model-worst j),
give KKT/water-fill, pin down when uniform is optimal, and add a CONVEX MINIMAX
allocation rule as a same-budget strong baseline, compared with uniform / neyman /
widthgreedy / sens / paid-CS on the existing four carriers.  Reuses the frozen
m3_cache artifacts (no retraining); only the reveal allocation changes.

Certificate mechanics are identical to M3/M4 (point-estimate selection + static
paired-MPB gate, delta=0.10, tau=0.04): only the per-group reveal count {n_g} differs.
Soundness is preserved because the water-fill allocation reads FIT-only statistics
(per-group cross-model paired-diff std), never CAL; reveal counts are fixed -> the
static MPB certificate remains valid (same argument as M3 prespecified rules).

The robust convex-minimax allocation (THEORY_MINIMAX_R1895.md sec 5):
  beta_g = per-group std across ordered model pairs of (fit_err_i - fit_err_j)
           on FIT held-out points of that group  (FIT-only, drives UCB width)
  eff_g(w) = w_g * beta_g
  water-fill  n_g ~ (eff_g)^(2/3)  with per-group floor and budget fill,
  where the ambiguity set over w in W is resolved by a fixed-point: start n=uniform,
  take w* = argmax_w sum_g w_g beta_g/sqrt(n_g) over the deployed grid, re-water-fill,
  iterate until n stable.  The 2/3 power is the KKT water-fill for 1/sqrt(n) width.

PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX  ROUND r1895  Pure CPU front / zero GPU.
"""
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # subgroup_mix_ranking/
sys.path.insert(0, os.path.join(ROOT, 'code'))
CACHE = os.path.join(ROOT, 'results', 'm3_cache')
OUT = os.path.join(ROOT, 'results', 'SUBGMIX_MINIMAX_R1895.json')

DELTA = 0.10
TAU = 0.04
FRACS = [0.50, 0.65, 0.80, 0.95]
SEEDS = [0, 1, 2]
CARRIERS = (sys.argv[1:] if len(sys.argv) > 1
            else ['digits', 'fashion', 'mnist', 'news'])


def load_art(name, seed):
    z = np.load(os.path.join(CACHE, f'subgmmix_m3_{name}_s{seed}.npz'), allow_pickle=True)
    Mnames = [str(m) for m in z['Mnames']]
    errs = {m: z[f'e_{m}'].astype(bool) for m in Mnames}
    fit_err = {m: z[f'fe_{m}'].astype(bool) for m in Mnames}
    outer = {m: {int(g): float(v) for g, v in z[f'outer_{m}'].item().items()} for m in Mnames}
    yf = z['yf'].astype(int); yc = z['yc'].astype(int)
    return errs, fit_err, outer, yf, yc, Mnames


def beta_from_fit(fit_err, yf, Mnames):
    """Per-group cross-model paired-diff std (FIT-only, drives UCB width)."""
    G = int(yf.max() + 1)
    fe = {m: fit_err[m].astype(float) for m in Mnames}
    beta = {}
    for g in range(G):
        sub = yf == g
        if sub.sum() < 2:
            beta[g] = 0.0
            continue
        sds = []
        for a in Mnames:
            for b in Mnames:
                if a == b:
                    continue
                sds.append(fe[a][sub].std(ddof=1))
        beta[g] = float(np.mean(sds)) if sds else 0.0
    return beta


def grid_from_recs(name):
    r = np.load(os.path.join(CACHE, f'recs_{name}_s0.npz'), allow_pickle=True)
    return r['grid'].item()


def _waterfill(eff, av, R, n_min):
    """Water-fill for a FIXED eff=w*beta: n_g = max(n_min, floor(eff_g^{2/3}*t)) with
    bisection on t, st sum = R, capped by avail; threshold groups with zero eff get n_min.
    Returns np array of n_g.  Budget remainder filled round-robin on largest fraction."""
    G = av.size
    big = eff ** (2.0 / 3.0)
    big = np.maximum(big, 1e-9)
    lo, hi = 1e-6, 1e12
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        cand = np.clip(np.minimum(np.floor(np.maximum(n_min, big * mid)), av), n_min, av)
        if cand.sum() <= R:
            lo = mid
        else:
            hi = mid
    n = np.clip(np.minimum(np.floor(np.maximum(n_min, big * lo)), av), n_min, av).astype(int)
    rem = int(R) - int(n.sum())
    order = np.argsort(-(big - n.astype(float)))
    i = 0
    while rem > 0:
        g = int(order[i % G])
        if n[g] < av[g]:
            n[g] += 1; rem -= 1
        i += 1
        if i % G == 0 and (n >= av).all():
            break
    return np.clip(n, n_min, av).astype(int)


def allocate_robustmm(beta, avail, R, grid, n_min=2, iters=6):
    """Fixed-point convex-minimax water-fill over the deployed ambiguity set W."""
    G = len(beta)
    b = np.array([beta[g] for g in range(G)])
    av = np.array([avail[g] for g in range(G)])
    n_min = int(min(n_min, av.max())) if av.size else n_min
    n_min = max(1, int(min(n_min, av.min()))) if av.size else 1
    W = np.array([[gp['w'][g] for g in range(G)] for gp in grid])  # (nW, G)
    n = np.clip(np.full(G, float(max(1, R // G))), 1, av)
    wstar = None
    for _ in range(iters):
        # worst deployed w given current n: maximize sum_g w_g beta_g/sqrt(n_g)
        score = W @ (b / np.sqrt(np.maximum(n, 1)))          # (nW,)
        wstar = W[np.argmax(score)]
        n = np.clip(_waterfill(wstar * b, av, R, n_min), n_min, av).astype(int)
    return {g: int(n[g]) for g in range(G)}, wstar


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


def run_carrier(name, seed):
    errs, fit_err, outer, yf, yc, Mnames = load_art(name, seed)
    G = int(yc.max() + 1)
    grid = grid_from_recs(name)
    beta = beta_from_fit(fit_err, yf, Mnames)
    avail = {g: int((yc == g).sum()) for g in range(G)}
    order = {}
    for g in range(G):
        idx = np.where(yc == g)[0]
        rng = np.random.RandomState(1000 + seed * 10 + g)
        order[g] = rng.permutation(idx).tolist()
    rows_all = {}
    for frac in FRACS:
        R = int(round(frac * len(yc)))
        R = min(R, sum(avail.values()))
        n, wstar = allocate_robustmm(beta, avail, R, grid)
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
    return G, grid, rows_all, beta, outer


def main():
    t0 = time.time()
    ACC = {}
    for name in CARRIERS:
        for seed in SEEDS:
            G, grid, rows_all, beta, outer = run_carrier(name, seed)
            n_w = len(grid)
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
                    'beta': [float(beta[g]) for g in range(G)]})
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
            'n_g_seed0': vals[0]['n_g'], 'beta_seed0': vals[0]['beta'],
            'abst_mean_regret': round(float(np.mean([v['abst_mean_regret'] for v in vals if v['abst_mean_regret'] is not None])), 4) if any(v['abst_mean_regret'] is not None for v in vals) else None,
            'abst_max_regret': round(float(np.max([v['abst_max_regret'] for v in vals if v['abst_max_regret'] is not None], initial=-1e9)), 4) if any(v['abst_max_regret'] is not None for v in vals) else None,
        }
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1895',
           'tau': TAU, 'delta': DELTA, 'agg': [agg[k] for k in sorted(agg)],
           'theory': 'THEORY_MINIMAX_R1895.md',
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