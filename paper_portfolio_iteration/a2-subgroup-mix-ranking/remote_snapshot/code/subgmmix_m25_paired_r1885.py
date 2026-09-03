"""M2.5: paired-difference (MCB-style) certificate for subgroup-mix ranking.

r1884 M2 gave cert_coverage=1.0 (sound) but committed_rate only 0.269, because
UB = U_{i*} - min_j L_j subtracts TWO *absolute* CP bands (median UB ~0.17 on
digits/news) while true regret is tiny (median 0.0000, max 0.0546).  The gate
is therefore gated by conservative absolute bands, not real model differences.

M2.5 fix (same problem, honest delta): the certificate target is
    regret(i*, w) = R_{i*}(w) - min_j R_j(w) = max_j (R_{i*}(w) - R_j(w)).
Each difference R_{i*}(w)-R_j(w) = sum_g w_g (r_{i*,g} - r_{j,g}) is a *paired*
within-CAL difference (same points, same group) -> shared error cancels, giving
a much tighter one-sided UCB than subtracting two independent absolute bands.
  - UCB_{i,j,g}: one-sided upper bound on the paired per-group difference
    d_{ijg}=mean_x[err_i(x)-err_j(x) | y=g],  unbiased for r_{i,g}-r_{j,g}.
  - Bonferroni over ALL ordered pairs (M(M-1)) x G groups => joint coverage
    >= 1-delta for every pair, so conditioning on the CAL-selected i* is safe
    (MCB / simultaneous best-comparison).
  - Certify i* at mixture w iff  max_{j!=i*} sum_g w_g UCB_{i*j,g} <= tau.
    On the joint event: regret(i*,w) <= tau  (sound).
Version of CI: normal-approx (CLT, n_g large, preferred, tightest) AND
Hoeffding-exact (range-2, guaranteed, wider) are both computed for comparison.
Soundness is verified empirically on OUTER (cert_coverage on committed rows).
PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX. ROUND: r1885.  Pure CPU / front / zero GPU.
"""
import json, time, numpy as np, os, sys
from scipy.stats import norm
sys.path.insert(0, 'subgroup_mix_ranking/code')
from subgmmix_m2_gate_r1884 import (load_carrier, load_news, w_grid, model_pool)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

OUT = "subgroup_mix_ranking/results/SUBGMIX_M25_PAIRED_R1885.json"
TAU = 0.04
CAL_FRAC = 0.30
SEEDS = [0, 1, 2]
DELTA = 0.10


def run_carrier(name, seed, use_which='normal'):
    X, y, kind = (load_news() if name == 'news' else load_carrier(name))
    G = int(y.max() + 1)
    X_all, X_outer, y_all, y_outer = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    pca = PCA(n_components=min(128, X_all.shape[1]), random_state=seed).fit(X_all)
    Z_all = pca.transform(X_all); Z_outer = pca.transform(X_outer)
    Z_fit, Z_cal, y_fit, y_cal = train_test_split(Z_all, y_all, test_size=CAL_FRAC, stratify=y_all, random_state=seed)
    pool = model_pool(seed); trained = {}
    for mid, m in pool.items():
        m.fit(Z_fit, y_fit); trained[mid] = m
    Mnames = list(trained.keys()); M = len(Mnames)
    # oracle (OUTER, diagnostics only)
    oracle = {}
    for mid, m in trained.items():
        pe = np.array(m.predict(Z_outer) != y_outer, dtype=float)
        oracle[mid] = {g: float(pe[y_outer == g].mean()) if (y_outer == g).sum() else 0.0 for g in range(G)}
    # per-model per-group point estimates (for selection)
    mist = {}
    for mid, m in trained.items():
        pred = m.predict(Z_cal); err = (pred != y_cal).astype(float)
        mist[mid] = {g: float(err[y_cal == g].mean()) if (y_cal == g).sum() else 0.0 for g in range(G)}
    # ONE-SIDED paired-difference UCB per ordered pair (i,j) and group g.
    dcell = DELTA / (M * (M - 1) * G)          # Bonferroni over ordered pairs x groups
    z = norm.ppf(1.0 - dcell)                   # normal one-sided
    hoef = np.sqrt(2.0 * np.log(1.0 / dcell))   # Hoeffding range-2 one-sided t
    UCB = {}                                    # UCB[(i,j)][g] = normal value
    HPW = {}                                    # Hoeffding value
    MPB = {}                                    # Maurer-Pontil empirical-Bernstein value
    for i in Mnames:
        erri = trained[i].predict(Z_cal) != y_cal
        for j in Mnames:
            if j == i:
                continue
            errj = trained[j].predict(Z_cal) != y_cal
            d = (erri.astype(float) - errj.astype(float))   # per-point, in {-1,0,1}
            for g in range(G):
                msk = y_cal == g
                n = int(msk.sum())
                if n == 0:
                    UCB[(i, j)][g] = None; HPW[(i, j)][g] = None; MPB[(i, j)][g] = None; continue
                dg = d[msk]; mu = dg.mean(); s = dg.std(ddof=1) if n > 1 else 0.0
                UCB.setdefault((i, j), {})[g] = mu + z * s / np.sqrt(n)
                HPW.setdefault((i, j), {})[g] = mu + hoef / np.sqrt(n)
                # Maurer-Pontil empirical-Bernstein, d in [-1,1] -> X=(d+1)/2 in [0,1],
                # var_X = s^2/4; one-sided: mu_X <= mX + sqrt(2 vX ln(2/d)/n) + 7 ln(2/d)/(3(n-1))
                if n > 1:
                    mX = (mu + 1.0) / 2.0
                    vX = (s ** 2) / 4.0
                    L = np.log(2.0 / dcell)
                    UCBX = mX + np.sqrt(2.0 * vX * L / n) + 7.0 * L / (3.0 * (n - 1))
                    MPB.setdefault((i, j), {})[g] = 2.0 * UCBX - 1.0
                else:
                    MPB.setdefault((i, j), {})[g] = mu + hoef / np.sqrt(n)
    u = {g: (y_cal == g).sum() / len(y_cal) for g in range(G)}
    rows = []
    for winfo in w_grid(G, u):
        w = winfo['w']; wname = winfo['name']
        trueR = {mid: sum(w[g] * oracle[mid][g] for g in range(G)) for mid in trained}
        bestR = min(trueR.values())
        ptR = {mid: sum(w[g] * mist[mid][g] for g in range(G)) for mid in trained}
        i = min(ptR, key=ptR.get)
        # M2.5 certificate: max over j of weighted paired UCB
        reg_bounds, reg_bounds_hoef, reg_bounds_mpb = {}, {}, {}
        for j in Mnames:
            if j == i: continue
            b = sum(w[g] * UCB[(i, j)][g] for g in range(G))
            bh = sum(w[g] * HPW[(i, j)][g] for g in range(G))
            bm = sum(w[g] * MPB[(i, j)][g] for g in range(G))
            reg_bounds[j] = b; reg_bounds_hoef[j] = bh; reg_bounds_mpb[j] = bm
        UB45 = max(reg_bounds.values()) if reg_bounds else 0.0
        UB45_hoef = max(reg_bounds_hoef.values()) if reg_bounds_hoef else 0.0
        UB45_mpb = max(reg_bounds_mpb.values()) if reg_bounds_mpb else 0.0
        committed = UB45 <= TAU
        reg = trueR[i] - bestR
        rows.append({'carrier': name, 'seed': seed, 'w': wname, 'chosen': i,
                     'true_regret': round(reg, 4), 'UB_paired': round(UB45, 4),
                     'UB_paired_hoef': round(UB45_hoef, 4), 'UB_paired_mpb': round(UB45_mpb, 4),
                     'committed': bool(committed),
                     'cert_regret': round(UB45 if committed else -1.0, 4)})
    return rows


def main():
    t0 = time.time(); all_rows = []
    for name in ['digits', 'fashion', 'mnist', 'news']:
        for seed in SEEDS:
            all_rows.extend(run_carrier(name, seed))
    n_comm = sum(1 for r in all_rows if r['committed'])
    comm_reg = [r['true_regret'] for r in all_rows if r['committed']]
    abs_reg = [r['true_regret'] for r in all_rows if not r['committed']]
    cov = np.mean([r <= TAU + 1e-9 for r in comm_reg]) if comm_reg else float('nan')
    # compare hoef committed (for exact variant)
    n_comm_hoef = sum(1 for r in all_rows if r['UB_paired_hoef'] <= TAU)
    comm_hoef_reg = [r['true_regret'] for r in all_rows if r['UB_paired_hoef'] <= TAU]
    cov_hoef = np.mean([r <= TAU + 1e-9 for r in comm_hoef_reg]) if comm_hoef_reg else float('nan')
    # MPB (Maurer-Pontil empirical-Bernstein) exact variant
    rows_mpb = [r for r in all_rows if r['UB_paired_mpb'] <= TAU]
    reg_mpb = [r['true_regret'] for r in rows_mpb]
    cov_mpb = np.mean([r <= TAU + 1e-9 for r in reg_mpb]) if reg_mpb else float('nan')
    agg = {'n_rows': len(all_rows), 'committed_rate_pair': round(n_comm / len(all_rows), 3),
           'cert_cov_pair': round(float(cov), 4), 'comm_mean_regret': round(float(np.mean(comm_reg)), 4) if comm_reg else None,
           'comm_max_regret': round(float(np.max(comm_reg)), 4) if comm_reg else None,
           'abst_mean_reg': round(float(np.mean(abs_reg)), 4) if abs_reg else None,
           'abst_max_reg': round(float(np.max(abs_reg)), 4) if abs_reg else None,
           'committed_rate_hoef': round(n_comm_hoef / len(all_rows), 3),
           'cert_cov_hoef': round(float(cov_hoef), 4),
           'committed_rate_mpb': round(len(rows_mpb) / len(all_rows), 3),
           'cert_cov_mpb': round(float(cov_mpb), 4),
           'tau': TAU, 'delta': DELTA, 'runtime_s': round(time.time() - t0, 1)}
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1885', 'agg': agg, 'rows': all_rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f: json.dump(out, f, indent=2)
    print(json.dumps(agg, indent=2)); print('saved', OUT)


if __name__ == '__main__':
    main()
