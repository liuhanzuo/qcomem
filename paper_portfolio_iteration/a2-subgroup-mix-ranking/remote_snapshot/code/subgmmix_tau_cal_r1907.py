"""M7 r1907: finite tau-menu, CAL-only tau selection under subgroup-mix turnover.

Builds directly on the frozen M2/M2.5/M6 pipeline (same split logic, same model_pool,
same normal paired-difference Bonferroni band) but on a FRESH, previously-unreported
disjoint seed block {5,6,7,8} (frozen evidence used seeds 0..4).

Question: if the operator must pick an acceptable per-point regret tolerance tau from
a finite menu T, can a CAL-only rule pick tau without paying extra multiplicity on the
certificate, and what does the selection cost?

Formalized in THEORY_TAU_CAL_R1907.md (Prop M7):
  - the frozen Bonferroni band is tau-INDEPENDENT (tau is only the comparison threshold
    in `committed iff D(i*,w) <= tau`); hence joint coverage >= 1-delta holds for ANY
    CAL-dependent tau_hat.  So selecting tau does not weaken the certificate.
  - the selection cost is purely performance-layer (which points commit, regret/coverage
    of committed points), bounded by (i) proxy mismatch + (ii) grid discretization +
    (iii) sampling variance, reported with paired intervals.

Arms (same budget: same CAL/OUTER, same band; only tau origin differs):
  1 fixed-tau  : commit iff D(i*,w)<=tau, per tau in T (0.04 = frozen default baseline)
  2 cal-select : tau_hat = min{tau in T : CR_cal(tau)>=p0}, empty -> min T (safe)
  3 sim-valid  : coverage of arm 2 on OUTER (should be ~1 by Prop M7)
  4 no-correction (naive): tau_naive = argmax CR_test(tau) reported on same test -> snooping
  5 oracle-tau : tau_oracle = best CR in hindsight (upper bound)

Endpoints (OUTER-settled): committed_rate, mean/max true_regret(committed),
coverage(true_regret<=tau), paired Delta vs fixed 0.04 (unit=seed x mixture, t*SEM CI),
SelCost = Perf(oracle) - Perf(cal), per-carrier weak domains.

PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX. ROUND r1907. Pure CPU / front / zero GPU.
One OUTER read for settlement ONLY (oracle regret), never to pick tau.
"""
import json, sys, numpy as np
from collections import defaultdict
from scipy.stats import norm, t as tdist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.insert(0, 'subgroup_mix_ranking/code')
from subgmmix_m2_gate_r1884 import load_carrier, load_news, model_pool, w_grid

TAU_MENU = [0.01, 0.02, 0.03, 0.04, 0.05]
P0 = 0.50
NEW_SEEDS = [5, 6, 7, 8]
CARRIERS = ['digits', 'fashion', 'mnist', 'news']
DELTA = 0.10
CAL_FRAC = 0.30
OUT = 'subgroup_mix_ranking/results/SUBGMIX_TAU_CAL_R1907.json'


def _D(UCB, i, w, Mnames, G):
    """D(i,w) = max_{j!=i} sum_g w_g UCB[(i,j)][g]; None iff no comparator."""
    best = -float('inf'); has = False
    for j in Mnames:
        if j == i:
            continue
        d = sum(w[g] * UCB[(i, j)][g] for g in range(G))
        best = max(best, d); has = True
    return best if has else None


def _cache_path(name, seed):
    return f'subgroup_mix_ranking/results/tau_cal_cache_{name}_s{seed}.pkl'


def build_cell(name, seed):
    """Replicate the frozen pipeline split / band / selection on a fresh seed.
    Caches the per-cell band/estimate/UCB struct via pickle so re-runs skip training."""
    import pickle
    cp = _cache_path(name, seed)
    try:
        with open(cp, 'rb') as f:
            cell = pickle.load(f)
        cell['carrier'] = name; cell['seed'] = seed
        return cell
    except Exception:
        pass
    cell = _build_cell_uncached(name, seed)
    with open(cp, 'wb') as f:
        pickle.dump(cell, f)
    print(f"  cached {name} s{seed}")
    cell['carrier'] = name; cell['seed'] = seed
    return cell


def _build_cell_uncached(name, seed):
    """Replicate the frozen pipeline split / band / selection on a fresh seed.
    Returns dict with trained models, CAL point estimates per model, UCB band,
    oracle (OUTER), u, F0, grid.  OUTER is read only for settlement."""
    X, y, _ = (load_news() if name == 'news' else load_carrier(name))
    G = int(y.max() + 1)
    X_all, X_outer, y_all, y_outer = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)
    pca = PCA(n_components=min(128, X_all.shape[1]), random_state=seed).fit(X_all)
    Z_all = pca.transform(X_all); Z_outer = pca.transform(X_outer)
    Z_fit, Z_cal, y_fit, y_cal = train_test_split(
        Z_all, y_all, test_size=CAL_FRAC, stratify=y_all, random_state=seed)
    pool = model_pool(seed); trained = {}
    for mid, m in pool.items():
        m.fit(Z_fit, y_fit); trained[mid] = m
    Mnames = list(trained.keys()); M = len(Mnames)

    oracle = {}                       # OUTER (settlement only)
    for mid, m in trained.items():
        pe = np.array(m.predict(Z_outer) != y_outer, dtype=float)
        oracle[mid] = {g: float(pe[y_outer == g].mean()) if (y_outer == g).sum() else 0.0
                       for g in range(G)}
    mist = {}                         # CAL point estimates (selection)
    for mid, m in trained.items():
        pred = m.predict(Z_cal); err = (pred != y_cal).astype(float)
        mist[mid] = {g: float(err[y_cal == g].mean()) if (y_cal == g).sum() else 0.0
                     for g in range(G)}

    dcell = DELTA / (M * (M - 1) * G) # tau-agnostic Bonferroni band
    z = norm.ppf(1.0 - dcell)
    UCB = {}
    for i in Mnames:
        erri = trained[i].predict(Z_cal) != y_cal
        for j in Mnames:
            if j == i:
                continue
            errj = trained[j].predict(Z_cal) != y_cal
            d = (erri.astype(float) - errj.astype(float))
            for g in range(G):
                msk = y_cal == g
                n = int(msk.sum())
                if n == 0:
                    UCB.setdefault((i, j), {})[g] = None; continue
                dg = d[msk]; mu = dg.mean(); s = dg.std(ddof=1) if n > 1 else 0.0
                UCB.setdefault((i, j), {})[g] = mu + z * s / np.sqrt(n)

    u = {g: (y_cal == g).sum() / len(y_cal) for g in range(G)}
    F0 = min(Mnames, key=lambda m: sum(u[g] * mist[m][g] for g in range(G)))
    grid = w_grid(G, u)
    return {'G': G, 'Mnames': Mnames, 'trained': trained, 'oracle': oracle,
            'mist': mist, 'UCB': UCB, 'u': u, 'F0': F0, 'grid': grid}


def settle(cell, winfo, tau):
    """OUTER-settled row for mixture winfo at a given tau.
    decision = i* if committed (D_i <= tau) else F0 (honest abstain fallback)."""
    G = cell['G']; Mnames = cell['Mnames']; w = winfo['w']
    ptR = {mid: sum(w[g] * cell['mist'][mid][g] for g in range(G)) for mid in Mnames}
    i = min(ptR, key=ptR.get)
    D = _D(cell['UCB'], i, w, Mnames, G)
    comm = bool(D is not None and D <= tau)
    dec = i if comm else cell['F0']
    trueR = {mid: sum(w[g] * cell['oracle'][mid][g] for g in range(G)) for mid in Mnames}
    bestR = min(trueR.values())
    reg = trueR[dec] - bestR
    return {'carrier': cell['carrier'], 'seed': cell['seed'], 'w': winfo['name'],
            'tau': tau, 'committed': comm, 'decision': dec, 'reg': reg,
            'reg_i': trueR[i] - bestR, 'D': (None if D is None else round(float(D), 4)),
            'i': i}


def cal_cr(cell):
    """CAL committed-rate per tau = empirical fraction of w-grid with certified D_i<=tau.
    Read only CAL band/estimates; never OUTER."""
    G = cell['G']; Mnames = cell['Mnames']
    Ds = []
    for winfo in cell['grid']:
        w = winfo['w']
        ptR = {mid: sum(w[g] * cell['mist'][mid][g] for g in range(G)) for mid in Mnames}
        i = min(ptR, key=ptR.get)
        d = _D(cell['UCB'], i, w, Mnames, G)
        if d is not None:
            Ds.append(d)
    return {tau: (float(np.mean([d <= tau for d in Ds])) if Ds else 0.0)
            for tau in TAU_MENU}


def agg_arm(rows):
    n = len(rows)
    committed = [r for r in rows if r['committed']]
    nc = len(committed); cr = nc / n
    regs = [r['reg'] for r in committed]
    mreg = float(np.mean(regs)) if regs else None
    mx = float(np.max(regs)) if regs else None
    cov = (float(np.mean([r['reg'] <= r['tau'] for r in committed])) if committed else None)
    return {'n': n, 'n_commit': nc, 'committed_rate': round(cr, 4),
            'mean_reg': (round(mreg, 4) if regs else None),
            'max_reg': (round(mx, 4) if regs else None),
            'coverage': (round(cov, 4) if committed else None)}


def main():
    cells = {}
    for name in CARRIERS:
        for seed in NEW_SEEDS:
            c = build_cell(name, seed)
            c['carrier'] = name; c['seed'] = seed
            cells[(name, seed)] = c
            print(f"built {name} seed{seed}: G={c['G']} grid={len(c['grid'])}")

    results = []
    for name in CARRIERS:
        for seed in NEW_SEEDS:
            c = cells[(name, seed)]
            crc = cal_cr(c)
            # CAL-only floor rule (pre-registered single rule)
            tau_hat = next((t for t in TAU_MENU if crc[t] >= P0), min(TAU_MENU))
            # evaluate fixed per-tau, cal-select, oracle, naive on OUTER (settlement)
            fixed = {t: [settle(c, winfo, t) for winfo in c['grid']] for t in TAU_MENU}
            cal_rows = [settle(c, winfo, tau_hat) for winfo in c['grid']]

            def perf(rows):
                cr_ = np.mean([r['committed'] for r in rows])
                regs = [r['reg'] for r in rows if r['committed']]
                mreg = float(np.mean(regs)) if regs else None
                return cr_, mreg
            # oracle-tau: best CR, tie-break lower committed mean regret (hindsight)
            cand = {t: perf(fixed[t]) for t in TAU_MENU}
            tau_oracle = max(TAU_MENU, key=lambda t: (cand[t][0], -(cand[t][1] if cand[t][1] is not None else 1e9)))
            # naive no-correction: pure CR snooping on OUTER
            tau_naive = max(TAU_MENU, key=lambda t: cand[t][0])
            oracle_rows = [settle(c, winfo, tau_oracle) for winfo in c['grid']]
            naive_rows = [settle(c, winfo, tau_naive) for winfo in c['grid']]
            results.append({'carrier': name, 'seed': seed, 'tau_hat': tau_hat,
                            'tau_oracle': tau_oracle, 'tau_naive': tau_naive,
                            'cal_cr': crc, 'cal': cal_rows, 'fixed': fixed,
                            'oracle': oracle_rows, 'naive': naive_rows})

    # ---- aggregate arms ----
    agg = {'n_carriers': len(CARRIERS), 'n_seeds': len(NEW_SEEDS), 'seeds': NEW_SEEDS,
           'tau_menu': TAU_MENU, 'p0': P0, 'delta': DELTA, 'cal_frac': CAL_FRAC,
           'arms': {}}
    for t in TAU_MENU:
        agg['arms'][f'fixed_tau_{t:.2f}'] = agg_arm([r for res in results for r in res['fixed'][t]])
    agg['arms']['cal_select'] = agg_arm([r for res in results for r in res['cal']])
    agg['arms']['oracle_tau'] = agg_arm([r for res in results for r in res['oracle']])
    agg['arms']['naive_no_correction'] = agg_arm([r for res in results for r in res['naive']])
    agg['tau_hat_dist'] = {str(t): int(sum(1 for r in results if r['tau_hat'] == t))
                           for t in TAU_MENU}

    # ---- paired Delta cal-select vs fixed 0.04 on reg (unit = seed x mixture) ----
    pairs = defaultdict(list)
    for res in results:
        cal_by = {(r['carrier'], r['seed'], r['w']): r for r in res['cal']}
        for r in res['fixed'][0.04]:
            km = (r['carrier'], r['seed'], r['w'])
            cr_ = cal_by.get(km)
            pairs[km].append({'cal_reg': cr_['reg'], 'fix_reg': r['reg']})
    dreg = np.array([p[0]['cal_reg'] - p[0]['fix_reg'] for p in pairs.values()])
    n_d = len(dreg); md = float(dreg.mean())
    se = float(dreg.std(ddof=1) / np.sqrt(n_d)) if n_d > 1 else None
    paired = {'n_units': n_d, 'mean_delta_reg_cal_minus_fixed004': round(md, 4),
              'sem': (round(se, 4) if se is not None else None)}
    if n_d > 1 and se is not None:
        tc = tdist.ppf(1 - DELTA / 2, n_d - 1)
        paired['ci095'] = [round(md - tc * se, 4), round(md + tc * se, 4)]
    agg['paired'] = paired

    mreg_c = agg['arms']['cal_select']['mean_reg']
    mreg_o = agg['arms']['oracle_tau']['mean_reg']
    agg['sel_cost'] = {'mean_reg_oracle_minus_cal':
                       (round(mreg_o - mreg_c, 4) if mreg_o is not None and mreg_c is not None else None),
                       'note': 'selection cost on performance layer only; certificate valid for either'}

    # ---- per-carrier weak-domain breakdown (cal arm + coverage) ----
    per_carrier = {}
    for name in CARRIERS:
        carows = [r for res in results if res['carrier'] == name for r in res['cal']]
        per_carrier[name] = agg_arm(carows)
        per_carrier[name]['tau_hat'] = [r['tau_hat'] for r in results if r['carrier'] == name]
        per_carrier[name]['cal_cr_range'] = {
            str(t): round(np.mean([res['cal_cr'][t] for res in results if res['carrier'] == name]), 4)
            for t in TAU_MENU}

    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1907',
           'note': 'finite tau-menu CAL-only selection (M7): band tau-independent; '
                   'simultaneous validity via Prop M7; cost is performance-only.',
           'agg': agg, 'per_carrier': per_carrier, 'results': results}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print("Wrote", OUT)
    print("AGG arms:")
    for k, v in agg['arms'].items():
        print("  ", k, json.dumps(v))
    print("tau_hat_dist:", agg['tau_hat_dist'])
    print("paired:", json.dumps(agg['paired']))
    print("sel_cost:", json.dumps(agg['sel_cost']))
    for c, v in per_carrier.items():
        print("  carrier", c, json.dumps(v))


if __name__ == '__main__':
    main()