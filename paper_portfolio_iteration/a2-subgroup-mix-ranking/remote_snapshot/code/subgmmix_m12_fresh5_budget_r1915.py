"""M12 r1915: fresh-5-seed real budget sweep of exact ABSOLUTE vs RELATIVE cert, M3.5 alloc.

MGR card f147b3bf33e9: on PRE-DECLARED fresh seeds, rerun REAL without-replacement
FIT / CAL / OUTER at the M3/M3.5 budget points (pi in {0.50,0.65,0.80,0.95}), and at the SAME
budget compare:
  (i) strictly finite-sample RELATIVE certificate (Hoeffding / MPB of the tau-free no-worse-
      than-F0 gate, M6/M8/M9/M10 object),
  (ii) EXACT ABSOLUTE certificate (M2.5 paired-difference gate, UB_paired <= TAU),
  (iii) status-quo F0 (what the operator would run anyway, no certificate),
  (iv) strong simple baselines: uniform spread and convex-minimax (M3.5 water-fill) allocation.
Report per carrier: commit rate, certificate coverage, real-switch (i*!=F0) admission, true
benefit/harm (mean & max OUTER REG_sq no-worse-than-F0), and compute cost.

This is the honest real-sampling companion to M8/M9/M10 on the FULL budget axis: M10 fixed a
per-group *label budget* b fraction of the CAL count; here the *M3/M3.5 label total* R=floor(pi*Ncal)
is re-split among subgroups by pre-specified rules reading FIT only, so the static paired-MPB/MPB
certificate is valid (no CAL-adaptive stopping).  Fresh seeds {10,11,12,13,14} are disjoint from
the frozen certificate seeds {0..4} and from the M7 CAL-selection seeds {5..8}.

Soundness of each committed row is settled on the held-out OUTER block (read once, never a gate
input).  All BPs and dcell follow the frozen M6/M8 pipeline (delta=0.1 collected-u w-grid, paired
single-sided, CAL_FRAC=0.30).

PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX.  ROUND: r1915.  Pure CPU / front / zero GPU.
"""
import json, time, os, sys, numpy as np
sys.path.insert(0, 'subgroup_mix_ranking/code')
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from subgmmix_m25_paired_r1885 import TAU, DELTA, CAL_FRAC
from subgmmix_m2_gate_r1884 import load_carrier, load_news, w_grid, model_pool
from subgmmix_m3_budget_r1886 import allocate                      # uniform/neyman/widthgreedy/sens
from subgmmix_minimax_r1895 import beta_from_fit, allocate_robustmm

FRESH = [10, 11, 12, 13, 14]
FRACS = [0.50, 0.65, 0.80, 0.95]
CARRIERS = ['digits', 'fashion', 'mnist', 'news']
OUT = "subgroup_mix_ranking/results/SUBGMIX_M12_FRESH5_BUDGET_R1915.json"


def bw_ucb(d, msk, g, dcell, variant):
    """Paired one-sided UCB for ordered pair diff in group g.  Same BPs as compute_bands(r1911)."""
    sub = d[msk]; n = int(sub.size)
    if n == 0:
        return None
    mu = sub.mean(); s = sub.std(ddof=1) if n > 1 else 0.0
    if variant == 'normal':
        return mu + norm.ppf(1.0 - dcell) * s / np.sqrt(n)
    if variant == 'hoef':
        return mu + np.sqrt(2.0 * np.log(1.0 / dcell)) / np.sqrt(n)
    # MPB (Bernstein variance term)
    z = norm.ppf(1.0 - dcell)
    hoef_c = np.sqrt(2.0 * np.log(1.0 / dcell))
    if n > 1:
        mX = (mu + 1.0) / 2.0; vX = (s ** 2) / 4.0; L = np.log(2.0 / dcell)
        return 2.0 * (mX + np.sqrt(2.0 * vX * L / n) + 7.0 * L / (3.0 * (n - 1))) - 1.0
    return mu + hoef_c / np.sqrt(n)


def settle(Mnames, cal_e, y_cal, rev, oracle, y_outer, grid, G, dcell, tau):
    """Reveal-setwise settlement: build paired bands on the revealed CAL rain, then per-w
    point-estimate selection + certificates; settle true REG_sq on OUTER (diagnostic only)."""
    yrev = y_cal[rev]
    # point-estimate per-group error on revealed CAL
    mist = {}
    for m in Mnames:
        e = cal_e[m][rev]
        mist[m] = {g: float(e[yrev == g].mean()) if (yrev == g).sum() else 0.0 for g in range(G)}
    uhat = {g: float((yrev == g).sum()) / max(1, rev.sum()) for g in range(G)}
    # collected-weight point estimate for F0 (status-quo optimal on collected mix)
    ptR_collect = {m: sum(uhat[g] * mist[m][g] for g in range(G)) for m in Mnames}
    F0 = min(ptR_collect, key=ptR_collect.get)
    # one-sided paired UCB on revealed CAL, per (i,j,g), each variant
    UCB = {}
    for var in ('normal', 'hoef', 'mpb'):
        UCB[var] = {}
        for i in Mnames:
            for j in Mnames:
                if i == j:
                    continue
                d = cal_e[i] - cal_e[j]
                for g in range(G):
                    v = bw_ucb(d, (y_cal == g) & rev, g, dcell, var)
                    UCB[var].setdefault((i, j), {})[g] = v
    rows = []
    for winfo in grid:
        w = winfo['w']; wname = winfo['name']
        # true OUTER risk (only read once, never a gate input)
        trueR = {m: sum(w[g] * float(oracle[m][y_outer == g].mean()) if (y_outer == g).sum() else 0.0
                        for g in range(G)) for m in Mnames}
        best = min(trueR.values())
        # point-estimate selection on revealed CAL at the DEPLOYMENT weight w
        ptRw = {m: sum(w[g] * mist[m][g] for g in range(G)) for m in Mnames}
        i = min(ptRw, key=ptRw.get)
        trivial = (i == F0)
        # --- exact absolute cert (M2.5): UB_paired <= tau ---
        reg_bounds = []
        for j in Mnames:
            if j == i:
                continue
            reg_bounds.append(sum(w[g] * (UCB['mpb'][(i, j)][g] or 1.0) for g in range(G)))
        UB_paired = max(reg_bounds) if reg_bounds else 0.0
        abs_commit = bool(UB_paired <= tau)
        # --- relative cert (M6): decide i vs F0 under exact bands ---
        dband = {}
        for var in ('normal', 'hoef', 'mpb'):
            dband[var] = 0.0 if i == F0 else float(sum(w[g] * (UCB[var][(i, F0)][g] or 0.0) for g in range(G)))
        dec = {}
        for var in ('normal', 'hoef', 'mpb'):
            cm = bool(dband[var] <= 0.0)
            dec[var] = (i if (cm and not trivial) else F0)
        row = {'w': wname, 'chosen': i, 'F0': F0, 'trivial': trivial,
               'true_regret_i': round(trueR[i] - best, 5), 'true_regret_F0': round(trueR[F0] - best, 5),
               'switch_gain_abs': round(trueR[F0] - trueR[i], 5),
               'abs_UB_paired': round(UB_paired, 4), 'abs_committed': abs_commit}
        for var in ('normal', 'hoef', 'mpb'):
            row[f'D_{var}'] = round(dband[var], 4)
            row[f'commit_{var}'] = bool(dband[var] <= 0.0)
            row[f'decision_{var}'] = dec[var]
            # REG_sq = true risk of decision vs F0 (benefit<0 / harm>0 from switching)
            row[f'REG_sq_{var}'] = round(trueR[dec[var]] - trueR[F0], 5)
        rows.append(row)
    return rows


def run_cell(name, seed):
    t0 = time.time()
    X, y, _ = (load_news() if name == 'news' else load_carrier(name))
    G = int(y.max() + 1)
    X_all, X_outer, y_all, y_outer = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    pca = PCA(n_components=min(128, X_all.shape[1]), random_state=seed).fit(X_all)
    Z_all = pca.transform(X_all); Z_outer = pca.transform(X_outer)
    Z_fit, Z_cal, y_fit, y_cal = train_test_split(Z_all, y_all, test_size=CAL_FRAC, stratify=y_all, random_state=seed)
    pool = model_pool(seed); trained = {}
    for mid, m in pool.items():
        m.fit(Z_fit, y_fit); trained[mid] = m
    Mnames = list(trained.keys()); M = len(Mnames)
    dcell = DELTA / (M * (M - 1) * G)
    fit_err = {m: (trained[m].predict(Z_fit) != y_fit).astype(float) for m in Mnames}
    cal_e = {m: (trained[m].predict(Z_cal) != y_cal).astype(float) for m in Mnames}
    oracle = {m: (trained[m].predict(Z_outer) != y_outer).astype(float) for m in Mnames}
    u = {g: int((y_cal == g).sum()) / max(1, len(y_cal)) for g in range(G)}
    grid = w_grid(G, u)
    avail = {g: int((y_cal == g).sum()) for g in range(G)}
    total_cal = int(len(y_cal))
    beta = beta_from_fit(fit_err, y_fit, Mnames)
    order = {}
    for g in range(G):
        rng = np.random.RandomState(2000 + seed * 10 + g)
        order[g] = rng.permutation(np.where(y_cal == g)[0]).tolist()
    rows_by = {}
    for frac in FRACS:
        R = int(round(frac * total_cal)); R = min(R, sum(avail.values()))
        pg_sd = {}
        for g in range(G):
            vals = [fit_err[m][y_fit == g].mean() if (y_fit == g).sum() else 0.0 for m in Mnames]
            pg_sd[g] = float(np.std(vals))
        scores = {'G': G, 'pg_sd': pg_sd, 'widthg': {g: 1.0 for g in range(G)}, 'worstW': u}
        for rule in ['uniform', 'neyman', 'sens', 'widthgreedy']:
            n = allocate(rule, scores, avail, R)
            rev = np.zeros(len(y_cal), bool)
            for g in range(G):
                if n[g] > 0:
                    rev[order[g][:n[g]]] = True
            rows_by[f"{frac}|{rule}"] = settle(Mnames, cal_e, y_cal, rev, oracle, y_outer, grid, G, dcell, TAU)
        n, _ = allocate_robustmm(beta, avail, R, grid)
        rev = np.zeros(len(y_cal), bool)
        for g in range(G):
            if n[g] > 0:
                rev[order[g][:n[g]]] = True
        rows_by[f"{frac}|convexminimax"] = settle(Mnames, cal_e, y_cal, rev, oracle, y_outer, grid, G, dcell, TAU)
    return {'G': G, 'n_full': {str(g): v for g, v in avail.items()}, 'runtime_s': round(time.time() - t0, 1),
            'rows_by': rows_by}


CELL_CACHE = "subgroup_mix_ranking/results/m12_cache/{name}_s{seed}.json"


def main():
    import json as _json
    t0 = time.time()
    os.makedirs(os.path.dirname(CELL_CACHE), exist_ok=True)
    cells = {}
    for name in CARRIERS:
        for seed in FRESH:
            ck = CELL_CACHE.format(name=name, seed=seed)
            if os.path.exists(ck):
                cells[(name, seed)] = _json.load(open(ck))
                print(f"  [{name} s{seed}] loaded from cache", flush=True)
                continue
            cells[(name, seed)] = run_cell(name, seed)
            with open(ck, 'w') as f:
                _json.dump(cells[(name, seed)], f)
            print(f"  [{name} s{seed}] done in {cells[(name, seed)]['runtime_s']}s", flush=True)
    # aggregate per (carrier, frac, rule)
    per_carrier = {}
    for name in CARRIERS:
        per_carrier[name] = {}
        for frac in FRACS:
            per_carrier[name][frac] = {}
            for rule in ['uniform', 'neyman', 'sens', 'widthgreedy', 'convexminimax']:
                rows = [r for s in FRESH for r in cells[(name, s)]['rows_by'][f"{frac}|{rule}"]]
                real = [r for r in rows if not r['trivial']]
                n = len(rows)
                g = {'n_rows': n,
                     'trivial_frac': round((n - len(real)) / n, 3),
                     'real_switch_count': len(real)}
                for var in ('normal', 'hoef', 'mpb'):
                    comm = [r for r in rows if r[f'commit_{var}']]
                    cr = round(len(comm) / n, 4) if n else None
                    regs = [r[f'REG_sq_{var}'] for r in comm] if comm else []
                    g[f'commit_{var}'] = cr
                    g[f'REG_sq_mean_{var}'] = round(float(np.mean(regs)), 5) if regs else None
                    g[f'REG_sq_max_{var}'] = round(float(np.max(regs)), 5) if regs else None
                    g[f'no_worse_cov_{var}'] = round(float(np.mean([r <= 1e-9 for r in regs])), 4) if regs else None
                comm_abs = [r for r in rows if r['abs_committed']]
                rabs = [r['true_regret_i'] for r in comm_abs]
                gains = [r['switch_gain_abs'] for r in comm_abs]           # + = switching away from F0 helps
                g['abs_commit'] = round(len(comm_abs) / n, 4) if n else None
                g['abs_truecov'] = round(float(np.mean([r <= TAU + 1e-9 for r in rabs])), 4) if rabs else None
                g['abs_gain_mean'] = round(float(np.mean(gains)), 5) if gains else None
                g['abs_gain_max'] = round(float(np.max(gains)), 5) if gains else None
                g['abs_gain_min'] = round(float(np.min(gains)), 5) if gains else None
                # genuine-switch admission under each exact relative band (the M10/M11 object)
                for var in ('hoef', 'mpb'):
                    realadm = [r for r in real if r[f'commit_{var}'] and not r['trivial']]
                    g[f'rel_real_admit_{var}'] = round(len(realadm) / max(1, len(real)), 4)
                    g[f'rel_real_admit_count_{var}'] = len(realadm)
                per_carrier[name][frac][rule] = g
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1915', 'meta': {
               'fresh_seeds': FRESH, 'fracs': FRACS, 'carriers': CARRIERS,
               'delta': DELTA, 'cal_frac': CAL_FRAC, 'tau_abs': TAU,
               'note': ('M12: fresh-5-seed real budget sweep on M3/M3.5 budget grid; '
                        'compare exact relative (hoef/mpb), exact absolute (M2.5), status-quo F0, '
                        'uniform + convex-minimax baselines; OUTER settled once (diagnostic).')},
           'per_carrier': per_carrier,
           'cells': {f"{n}_{s}": {'G': cl['G'], 'n_full': cl['n_full'], 'runtime_s': cl['runtime_s'],
                                  'rows_by': cl['rows_by']} for (n, s), cl in cells.items()},
           'runtime_total_s': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    # console readback
    print("n real-switch rows (full split, all frac) & exact-relative admission per carrier:")
    for name in CARRIERS:
        line = [name]
        for frac in FRACS:
            for rule in ['uniform', 'convexminimax']:
                g = per_carrier[name][frac][rule]
                line.append(f"pi{frac}:{rule[0]}=cr{rule[0]}{g['commit_hoef']}/abs{g['abs_commit']}"
                            f"real{g['real_switch_count']}")
        print("  ", ' | '.join(line))
    print('saved', OUT, ' runtime_total_s', out['runtime_total_s'])
    sys.exit(0)


if __name__ == '__main__':
    main()