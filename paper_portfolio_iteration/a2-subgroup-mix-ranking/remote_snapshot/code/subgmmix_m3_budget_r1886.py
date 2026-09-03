"""M3: budgetized adaptive/pre-specified subgroup allocation for safe ranking (r1886).

Same problem as r1884 (M2) / r1885 (M2.5): finite-sample safe model ranking under
subgroup-mix turnover.  M3 adds the LABEL-BUDGET axis + subgroup ALLOCATION.

Setup (mirrors M2.5 so results are comparable):
  - Fixed FIT/CAL split.  FIT labels are already known (training data); CAL labels are budgeted.
  - Selection: point-estimate argmin_i pt_i(w) (CI only a gate, never a selector: M1 lesson).
  - Certificate (M2.5 paired-difference, MCB-style): for i*(w),
        UB(w)=max_j sum_g w_g UCB_{i*j g}; committ iff UB(w)<=tau (regret<=tau sound).
  - NEW M3: an allocation rule maps a total CAL-label budget R -> per-group reveal counts
    {n_g: sum=R}.  Sweep R across the budget axis (FRACS of total CAL), and at each R compare
    5 rules at the SAME total budget.  cert_validity and committed_rate STRICTLY SEPARATED.

Soundness (key correctness point):
  - PRE-SPECIFIED rules (uniform / neyman / widthgreedy / sens) decide {n_g} using ONLY the
    already-labeled FIT split.  Reveal counts are FIXED (not a CAL-data-dependent stopping
    time), so the standard M2.5 static paired-MPB certificate is valid.  This is budgetized
    certificate tightening with a tight sound finite-sample certificate.
  - FULLY-ADAPTIVE rule (adaptive) reads CAL labels one at a time to steer the next reveal ->
    data-dependent stopping time; static MPB certificate would be INVALID.  We certify this
    track with a TIME-UNIFORM Hoeffding confidence sequence (always-valid at every count).
    The extra width is the certificate-price of adaptivity; we report it honestly.
  - Abstained rows go to an explicit point-estimate fallback; their TRUE OUTER regret is
    reported (abst_*), and hardpick (always argmin pt) is the negative control.  We never
    count abstain as a certificate gain.

PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX  ROUND: r1886  Pure CPU / front / zero GPU.
"""
import json, time, numpy as np, os, sys
sys.path.insert(0, 'subgroup_mix_ranking/code')
from subgmmix_m2_gate_r1884 import load_carrier, load_news, w_grid, model_pool
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

OUT = "subgroup_mix_ranking/results/SUBGMIX_M3_BUDGET_R1886.json"
TAU_LIST = [0.02, 0.04, 0.06]
FRACS = [0.50, 0.65, 0.80, 0.95]
SEEDS = [0, 1, 2]
DELTA = 0.10
CARRIERS = (sys.argv[1:] if len(sys.argv) > 1
            else ['digits', 'fashion', 'mnist', 'news'])


def dcell_static(M, G):
    return DELTA / (M * (M - 1) * max(G, 1))


def dcell_cs(M, G):
    return DELTA / (M * (M - 1) * max(G, 1))


def build_ucb_mpb(errs, y_cal, rev, Mnames, G, dcell):
    """Static paired-MPB UCB (valid when reveal is non-data-dependent on CAL)."""
    UCB = {}
    for i in Mnames:
        erri = errs[i].astype(float)
        for j in Mnames:
            if i == j:
                continue
            d = erri - errs[j].astype(float)
            for g in range(G):
                sub = d[(y_cal == g) & rev]; n = int(sub.size)
                if n == 0:
                    UCB.setdefault((i, j), {})[g] = None
                    continue
                mu = sub.mean(); s = sub.std(ddof=1) if n > 1 else 0.0
                mX = (mu + 1.0) / 2.0; vX = (s ** 2) / 4.0; L = np.log(2.0 / dcell)
                if n > 1:
                    ubX = mX + np.sqrt(2.0 * vX * L / n) + 7.0 * L / (3.0 * (n - 1))
                else:
                    ubX = mX + np.sqrt(L / 2.0)
                UCB.setdefault((i, j), {})[g] = 2.0 * ubX - 1.0
    return UCB


def build_ucb_cs(errs, y_cal, rev, Mnames, G):
    """Time-uniform empirical-Bernstein confidence sequence (always-valid at every count).
    delta_n split by 6/(pi^2 n (n+1)) so union <= dcell.  Valid for ANY adaptive stopping time
    on the reveal set.  Bennett-style variance-shrunk bound on d in [-1,1]:
    mean m=(mu+1)/2 in [0,1], per-sample variance v/4, mu_ub = 2*[m + sqrt(2 (v/4) b / n) + 7b/(3(n-1))] - 1.
    (MGR 9e60b561dcd7: "用 union-valid empirical-Bernstein/置信序列处理自适应采样".)
    """
    dc = dcell_cs(len(Mnames), G)
    UCB = {}
    for i in Mnames:
        erri = errs[i].astype(float)
        for j in Mnames:
            if i == j:
                continue
            d = erri - errs[j].astype(float)
            for g in range(G):
                sub = d[(y_cal == g) & rev]; n = int(sub.size)
                if n == 0:
                    UCB.setdefault((i, j), {})[g] = None
                    continue
                dn = dc * 6.0 / (np.pi ** 2 * n * (n + 1)) if n >= 1 else dc
                b = np.log(2.0 / dn)
                m = (sub.mean() + 1.0) / 2.0
                v = (sub.var(ddof=1) / 4.0) if n > 1 else 0.0
                ubX = m + np.sqrt(2.0 * v * b / n) + 7.0 * b / (3.0 * (n - 1)) if n > 1 \
                    else m + np.sqrt(b / 2.0)
                UCB.setdefault((i, j), {})[g] = 2.0 * ubX - 1.0
    return UCB


def selection_and_cert(errs, y_cal, rev, UCB, Mnames, G, w):
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


CACHE_DIR = "subgroup_mix_ranking/results/m3_cache/"


def get_artifacts(name, seed):
    """Train models once per (carrier,seed), cache fitted predictions, return (errs, fit_err,
    outer, yf, yc).  Cache lets a turn-boundary-interrupted M3 run resume without retraining
    fashion/mnist (~2500s/seed)."""
    key = os.path.join(CACHE_DIR, f"subgmmix_m3_{name}_s{seed}.npz")
    if os.path.exists(key):
        z = np.load(key, allow_pickle=True)
        Mnames = list(z['Mnames'])
        outer = {m: {int(g): float(v) for g, v in z[f'outer_{m}'].item().items()} for m in Mnames}
        return ({m: z[f'e_{m}'].astype(bool) for m in Mnames},
                {m: z[f'fe_{m}'].astype(bool) for m in Mnames},
                outer, z['yf'].astype(int), z['yc'].astype(int))
    X, y, _ = (load_news() if name == 'news' else load_carrier(name))
    G = int(y.max() + 1)
    Xa, Xo, ya, yo = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    pca = PCA(n_components=min(128, Xa.shape[1]), random_state=seed).fit(Xa)
    Za = pca.transform(Xa); Zo = pca.transform(Xo)
    Zf, Zc, yf, yc = train_test_split(Za, ya, test_size=0.30, stratify=ya, random_state=seed)
    pool = model_pool(seed); Mnames = list(pool.keys())
    errs = {}
    for mid, mo in pool.items():
        mo.fit(Zf, yf)
        errs[mid] = (mo.predict(Zc) != yc)
    outer = {}
    for mid, mo in pool.items():
        oe = (mo.predict(Zo) != yo).astype(float)
        outer[mid] = {g: float(oe[yo == g].mean()) if (yo == g).sum() else 0.0 for g in range(G)}
    # FIT-derived scores for pre-specified allocation (uses already-labeled FIT, no CAL peek)
    fit_err = {mid: (pool[mid].predict(Zf) != yf).astype(float) for mid in Mnames}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(key, yf=yf, yc=yc, Mnames=np.array(Mnames, dtype=object),
             **{f'e_{m}': errs[m] for m in Mnames},
             **{f'fe_{m}': fit_err[m] for m in Mnames},
             **{f'outer_{m}': np.array(outer[m]) for m in Mnames})
    return errs, fit_err, outer, yf, yc


REC_CACHE = "subgroup_mix_ranking/results/m3_cache/recs_{name}_s{seed}.npz"


def run_carrier(name, seed):
    # Resumable: if per-(carrier,seed) reveal rows already cached, load them instead of
    # recomputing (lets a turn-boundary interrupted M3 run resume for free).
    rk = REC_CACHE.format(name=name, seed=seed)
    if os.path.exists(rk):
        z = np.load(rk, allow_pickle=True)
        G = int(z['G'].item()); grid = z['grid'].item()
        recs = {tuple(k): v for k, v in z['recs'].item().items()}
        outer = z['outer'].item()
        outer = {m: {int(g): float(v) for g, v in op_.item().items()} for m, op_ in outer.items()}
        return G, grid, recs, outer
    errs, fit_err, outer, yf, yc = get_artifacts(name, seed)
    G = int(max(int(yf.max()), int(yc.max())) + 1)
    Mnames = list(errs.keys())
    pg_mu, pg_sd = {}, {}
    for g in range(G):
        vals = [fit_err[i][yf == g].mean() if (yf == g).sum() else 0.0 for i in Mnames]
        pg_mu[g] = float(np.mean(vals)); pg_sd[g] = float(np.std(vals)) if len(vals) >= 2 else 0.0
    u = {g: (yf == g).sum() / len(yf) for g in range(G)}
    wgrid = w_grid(G, u)
    widthg = {g: float(pg_sd[g]) + 1e-9 for g in range(G)}
    ptR = {mid: sum(u[g] * (fit_err[mid][yf == g].mean() if (yf == g).sum() else 0.0) for g in range(G)) for mid in Mnames}
    i0 = min(ptR, key=ptR.get)
    worstW = None; wv = -1e9
    for wf in wgrid:
        r = sum(wf['w'][g] * pg_mu[g] for g in range(G))
        if r > wv:
            wv = r; worstW = wf['w']
    if worstW is None:
        worstW = u
    scores = {'pg_sd': pg_sd, 'widthg': widthg, 'worstW': worstW, 'G': G,
              'pg_mu': pg_mu, 'u': u, 'Mnames': Mnames}
    # CAL grid under collected-u weights (same as M2.5 evaluation)
    ucal = {g: (yc == g).sum() / len(yc) for g in range(G)}
    grid = w_grid(G, ucal)
    avail = {g: int((yc == g).sum()) for g in range(G)}
    total_cal = int(len(yc))
    order = {g: [] for g in range(G)}
    for g in range(G):
        idx = np.where(yc == g)[0]
        rng = np.random.RandomState(1000 + seed * 10 + g)
        order[g] = rng.permutation(idx).tolist()
    recs = {}
    for frac in FRACS:
        R = int(round(frac * total_cal))
        R = min(R, sum(avail.values()))
        for policy in ['uniform', 'neyman', 'widthgreedy', 'sens']:
            n = allocate(policy, scores, avail, R)
            rev = np.zeros(len(yc), bool)
            for g in range(G):
                if n[g] > 0:
                    rev[order[g][:n[g]]] = True
            dc = dcell_static(len(Mnames), G)
            UCB = build_ucb_mpb(errs, yc, rev, Mnames, G, dc)
            rows = []
            for winfo in grid:
                i, ub = selection_and_cert(errs, yc, rev, UCB, Mnames, G, winfo['w'])
                trueR = {mid: sum(winfo['w'][g] * outer[mid][g] for g in range(G)) for mid in Mnames}
                reg = trueR[i] - min(trueR.values())
                rows.append({'w': winfo['name'], 'chosen': i, 'UB': float(ub),
                             'true_regret': float(reg), 'label_cost': int(rev.sum()), 'cert_type': 'mpb'})
            recs[(frac, policy)] = rows
        # FULLY-ADAPTIVE (CS track, always-valid): one-at-a-time steer on worst vertex width
        rev_o = np.zeros(len(yc), bool); cnt = {g: 0 for g in range(G)}
        # pre-seed 1 core per group so every group has n>=1 (no None UCB)
        used = 0
        for g in range(G):
            if avail[g] > 0:
                rev_o[order[g][0]] = True; cnt[g] = 1; used += 1
        while used < R:
            anyav = sum(1 for g in range(G) if cnt[g] < avail[g])
            if anyav == 0:
                break
            UCBc = build_ucb_cs(errs, yc, rev_o, Mnames, G)
            worst = None; wv = -1e9
            for winfo in grid:
                _, ub = selection_and_cert(errs, yc, rev_o, UCBc, Mnames, G, winfo['w'])
                if ub > wv:
                    wv = ub; worst = winfo
            if worst is None:
                break
            sub_avail = [g for g in range(G) if cnt[g] < avail[g]]
            if not sub_avail:
                break
            iw, _ = selection_and_cert(errs, yc, rev_o, UCBc, Mnames, G, worst['w'])
            gw = {}
            for g in range(G):
                m = -1e9
                for j in Mnames:
                    if j == iw:
                        continue
                    v = UCBc[(iw, j)][g]
                    m = max(m, v if v is not None else -1e9)
                gw[g] = m
            def key(g):
                return (worst['w'][g] * gw[g]) if cnt[g] < avail[g] else -1e9
            g = max(range(G), key=key)
            if cnt[g] >= avail[g]:
                g = max(sub_avail, key=lambda gg: (avail[gg] - cnt[gg]))
            rev_o[order[g][cnt[g]]] = True; cnt[g] += 1; used += 1
        UCBc = build_ucb_cs(errs, yc, rev_o, Mnames, G)
        rows = []
        for winfo in grid:
            i, ubc = selection_and_cert(errs, yc, rev_o, UCBc, Mnames, G, winfo['w'])
            trueR = {mid: sum(winfo['w'][g] * outer[mid][g] for g in range(G)) for mid in Mnames}
            reg = trueR[i] - min(trueR.values())
            rows.append({'w': winfo['name'], 'chosen': i, 'UB': float(ubc),
                         'true_regret': float(reg), 'label_cost': int(rev_o.sum()), 'cert_type': 'cs'})
        recs[(frac, 'adaptive')] = rows
    # Checkpoint per-(carrier,seed) so an interrupted multi-carrier run resumes for free.
    os.makedirs(os.path.dirname(rk), exist_ok=True)
    def _wrap(x):
        a = np.empty(1, dtype=object); a[0] = x; return a  # force shape (1,) object
    np.savez(rk, G=_wrap(G), grid=_wrap(grid), recs=_wrap(recs),
             outer=_wrap({m: np.array(op_) for m, op_ in outer.items()}))
    return G, grid, recs, outer


def allocate(policy, scores, avail, R):
    G = scores['G']
    if policy == 'uniform':
        w = np.array([1.0] * G)
    elif policy == 'neyman':
        w = np.array([scores['pg_sd'][g] for g in range(G)]) + 1e-9
    elif policy == 'widthgreedy':
        w = np.array([scores['widthg'][g] for g in range(G)])
    elif policy == 'sens':
        w = np.array([scores['worstW'][g] * scores['widthg'][g] for g in range(G)])
    else:
        raise ValueError(policy)
    w = np.maximum(w, 1e-9)
    av = np.array([avail[g] for g in range(G)])
    raw = w / w.sum() * R
    n = np.minimum(np.floor(raw), av).astype(int)
    rem = int(R) - int(n.sum())
    # Full remainder fill (round-robin on largest fraction first, repeat until R spent or
    # all groups capped).  Single-pass breaks imbalanced allocations (sens) which floor to
    # almost nothing and then waste the budget -> unfair comparison at the same total R.
    order = np.argsort(-(raw - n))
    i = 0
    while rem > 0:
        g = order[i % len(order)]
        if n[g] < av[g]:
            n[g] += 1; rem -= 1
        i += 1
        if i % len(order) == 0 and (n >= av).all():
            break
    return {g: int(n[g]) for g in range(G)}


def main():
    t0 = time.time()
    ACC = {}   # key -> list of per-seed metric dicts (aggregated honestly over seeds)
    for name in CARRIERS:
        for seed in SEEDS:
            G, grid, recs, _ = run_carrier(name, seed)
            n_w = len(grid)
            for (frac, policy), rows in recs.items():
                for tau in TAU_LIST:
                    key = (name, frac, policy, tau)
                    comm = [r for r in rows if r['UB'] <= tau]
                    cr = len(comm) / n_w if n_w else 0.0
                    labels = np.mean([r['label_cost'] for r in rows]) if rows else 0.0
                    if comm:
                        regs = np.array([r['true_regret'] for r in comm])
                        cv = float(np.mean(regs <= tau + 1e-9))
                    else:
                        cv = float('nan')
                    abst = [r for r in rows if r['UB'] > tau]
                    amean = float(np.mean([r['true_regret'] for r in abst])) if abst else None
                    amax = float(np.max([r['true_regret'] for r in abst])) if abst else None
                    ACC.setdefault(key, []).append(
                        {'carrier': name, 'frac': frac, 'policy': policy, 'tau': tau,
                         'seed': seed, 'R_labels': int(round(labels)), 'n_w': n_w,
                         'committed_rate': cr,
                         'cert_validity': cv if cv == cv else None,
                         'abst_mean_regret': amean if amean is not None else None,
                         'abst_max_regret': amax if amax is not None else None,
                         'n_commit': len(comm)})
    # Aggregate over seeds: mean committed_rate / exposure-weighted cert validity / max abst regret.
    agg = {}
    for key, vals in ACC.items():
        crs = [v['committed_rate'] for v in vals]
        nw = vals[0]['n_w']
        n_commit_tot = sum(v['n_commit'] for v in vals)
        cvs = [v['cert_validity'] for v in vals if v['cert_validity'] is not None]
        cv_agg = float(np.mean(cvs)) if cvs else None
        abst_means = [v['abst_mean_regret'] for v in vals if v['abst_mean_regret'] is not None]
        abst_maxes = [v['abst_max_regret'] for v in vals if v['abst_max_regret'] is not None]
        agg[key] = {'carrier': vals[0]['carrier'], 'frac': vals[0]['frac'],
                    'policy': vals[0]['policy'], 'tau': vals[0]['tau'],
                    'R_labels': int(np.mean([v['R_labels'] for v in vals])),
                    'n_w': nw, 'n_seeds': len(vals),
                    'committed_rate': round(float(np.mean(crs)), 3),
                    'cert_validity': round(cv_agg, 4) if cv_agg is not None else None,
                    'abst_mean_regret': round(float(np.mean(abst_means)), 4) if abst_means else None,
                    'abst_max_regret': round(float(np.max(abst_maxes, initial=-1e9)), 4) if abst_maxes else None}
    agg_list = [agg[k] for k in sorted(agg, key=lambda k: (k[0], k[1], k[3]))]
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1886',
           'agg': agg_list, 'meta': {'TAU_LIST': TAU_LIST, 'FRACS': FRACS, 'SEEDS': SEEDS,
                                     'DELTA': DELTA, 'CARRIERS': CARRIERS},
           'runtime_s': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('runtime_s', out['runtime_s'])
    print('committed_rate @ cert_validity, tau=0.04:')
    for name in CARRIERS:
        for frac in FRACS:
            line = [f"{name} frac={frac}"]
            for policy in ['uniform', 'neyman', 'widthgreedy', 'sens', 'adaptive']:
                if (name, frac, policy, 0.04) in agg:
                    a = agg[(name, frac, policy, 0.04)]
                    line.append(f"{policy}={a['committed_rate']}(cv{a['cert_validity']})")
            print('  ', ' | '.join(line))
    print('saved', OUT)


if __name__ == '__main__':
    main()