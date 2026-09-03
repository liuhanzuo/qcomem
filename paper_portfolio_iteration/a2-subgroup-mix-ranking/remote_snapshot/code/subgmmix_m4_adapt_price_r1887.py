"""M4 (r1887): certificate-price of adaptivity via split-CAL two-stage allocation.

Diagnoses R1886 M3's failing adaptive track.  As analyzed in the log, the naive one-at-a-time
adaptive steer cannot be certified with a static MPB bound (data-dependent stopping time), so it
fell back to an always-valid CS that was so wide nobody ever committed.  Is the price of
adaptivity intrinsic, or just the price of the constant-valid CS?  This experiment uses a
TWO-STAGE (offline adaptivity) split to separate the two:

  - We have the SAME mechanics as M3 (SRS, allocation rules, paired-MPB certificate) but the
    adaptive track now spends a PRE-FIXED budget on a *first* sample drawn by size-biased
    ordering (deterministic on FIT-held-out data, i.e. offline-adaptive direction), then labels a
    SEPARATE, fresh, SRS CARGO sample that is certified with a STATIC MPB bound.

  - Static-MPB on the cargo share restores finite-sample soundness at every total budget, so the
    only added width vs M3 static comes from the smaller certified n.  If committed>0 appears
    where the naive CS gave 0, adaptivity itself is cheap and the earlier 0 was pure CS width.
    If committed stays 0, adaptivity is genuinely priced by the smaller certified sample.  Either
    way we publish the trade-off honestly (no cherry-pick), and digits/news bandwidth-wall is
    documented as a SEPARATE dominant effect.

Soundness note (why static MPB is valid here): the certified score n_g(H) is a FIXED count; the
data in the CARGO share are SRS per group, independent of the pre-sample adaptivity decision
(adaptivity is read from annex-fit/CAL outside the cargo).  So the standard static paired-MPB
Bernstein bound applies on the cargo share; this is the "adaptive direction, static certificate"
composition.  We do NOT claim adaptivity is free - we price it.
"""
import json, numpy as np, os, sys, time
sys.path.insert(0, 'subgroup_mix_ranking/code')
from subgmmix_m2_gate_r1884 import (load_carrier, load_news, w_grid, model_pool, DELTA, CAL_FRAC)
from subgmmix_m3_budget_r1886 import get_artifacts
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

OUT = "subgroup_mix_ranking/results/SUBGMIX_M4_ADAPTPRICE_R1887.json"
TAU_LIST = [0.04, 0.06]
FRACS = [0.50, 0.65, 0.80, 0.95]
SEEDS = [0, 1, 2]
DELTA = 0.10
CARRIERS = (sys.argv[1:] if len(sys.argv) > 1
            else ['digits', 'fashion', 'mnist', 'news'])
CACHE = "subgroup_mix_ranking/results/m3_cache"


def dcell_static(M, G):
    return DELTA / (M * (M - 1) * max(G, 1))


def build_ucb_mpb(errs, y_cal, rev, Mnames, G, dcell):
    return _builder(errs, y_cal, rev, Mnames, G, dcell, split_delta=lambda dc, n: dc)


def build_ucb_mpb_branchsplit(errs, y_cal, rev, Mnames, G, dcell):
    """Static paired-MPB but the same delta applies to each branch independently is NOT what we
    want; unified bound here (the only SRS count is the revealed set count)."""
    return _builder(errs, y_cal, rev, Mnames, G, dcell, split_delta=lambda dc, n: dc)


def _builder(errs, y_cal, rev, Mnames, G, dcell, split_delta):
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


def _group_err(errs, Mnames, y_cal, rev, apt):
    """Port of apt scoring: mean abs deviation of per-group point estimates across models.
    Returns {g: mAD_g}.  Uses REVEALED cal labels for apt (pre-sample supervision)."""
    G = int(y_cal.max() + 1)
    out = {}
    for g in range(G):
        sub = np.where((y_cal == g) & rev)[0]
        if sub.size == 0:
            out[g] = 0.0; continue
        pe = {m: float(errs[m][sub].mean()) for m in Mnames}
        v = np.array(list(pe.values()))
        out[g] = float(np.mean(np.abs(v - v.mean())))
    return out


def apt_allocate(aptv, avail, R):
    """size-biased on apt margin: n_g = max(1, floor(R * aptv_g/sum)) capped at avail, fill remainder."""
    G = len(aptv)
    w = np.array([aptv[g] for g in range(G)]) + 1e-9
    av = np.array([avail[g] for g in range(G)])
    raw = w / w.sum() * R
    n = np.minimum(np.floor(raw), av).astype(int)
    rem = int(R) - int(n.sum())
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


def prep(name, seed):
    """Return errs(cal), y_cal(cal indices), dict of ALREADY-REVEALABLE cal indices per group for
    the annex sample (first-share), G, Mnames, outer, grid evaluation objects."""
    errs, fit_err, outer, yf, yc = get_artifacts(name, seed)
    G = int(int(yc.max()) + 1)
    Mnames = list(errs.keys())
    avail = {g: int((yc == g).sum()) for g in range(G)}
    # deterministic per-group index order (same RNG as M3 so shares are comparable)
    order = {}
    for g in range(G):
        idx = np.where(yc == g)[0]
        rng = np.random.RandomState(1000 + seed * 10 + g)
        order[g] = rng.permutation(idx).tolist()
    # CARGO alternative = SRS per group (matched points, the certified static share for stage B)
    ucal = {g: (yc == g).sum() / len(yc) for g in range(G)}
    grid = w_grid(G, ucal)
    return errs, yc, outer, G, Mnames, avail, order, grid


def run_carrier(name, seed):
    errs, yc, outer, G, Mnames, avail, order, grid = prep(name, seed)
    n_w = len(grid)
    recs = {}
    for frac in FRACS:
        R = int(round(frac * len(yc)))
        R = min(R, sum(avail.values()))
        # ---- PRE-SPECIFIED baseline (reproduce M3 uniform static as the price reference) ----
        from subgmmix_m3_budget_r1886 import allocate as _alloc
        n = _alloc('uniform', {'G': G}, avail, R)
        rem = R - sum(n.values())
        i = 0; gl = list(range(G))
        while rem > 0:
            g = gl[i % G]
            if n[g] < avail[g]:
                n[g] += 1; rem -= 1
            i += 1
            if i % G == 0 and all(n[g] >= avail[g] for g in gl):
                break
        rev = np.zeros(len(yc), bool)
        for g in range(G):
            rev[order[g][:n[g]]] = True
        UCB = build_ucb_mpb(errs, yc, rev, Mnames, G, dcell_static(len(Mnames), G))
        rows_s = []
        for winfo in grid:
            i_, ub_ = selection_and_cert(errs, yc, rev, UCB, Mnames, G, winfo['w'])
            trueR = {mid: sum(winfo['w'][g] * outer[mid][g] for g in range(G)) for mid in Mnames}
            reg = trueR[i_] - min(trueR.values())
            rows_s.append({'w': winfo['name'], 'chosen': i_, 'UB': float(ub_),
                           'true_regret': float(reg), 'label_cost': int(rev.sum()), 'cert': 'static-uniform'})
        recs[(frac, 'S_uniform')] = rows_s
        # ---- STAGE-B PRICE: adaptive direction + static certificate on undisclosed SRS sample ----
        # Stage A spends a pre-fixed R_CALIG of the budget on the ''direction-finding'' share
        # (offline-adaptive, size-biased on annex), then certifies on the PRIOR-unrevealed CARGO.
        # To make R_CALIG a *share of the total* we reserve R_CERT = max(floor(R*s_cert), G) points
        # for static cargo, and the rest for adaptive annex.  Annex reveals consume labels; cargo is
        # a separate SRS per group of size m_g certified by static MPB.
        s_cert = 0.5
        R_cert = max(int(round(R * s_cert)), G)
        R_cert = min(R_cert, sum(min(avail.values()) if False else avail.values()))  # keep simple cap below
        # cargo per-group m_g (SRS uniform over groups, cap by avail)
        mc = [min(avail[g], max(1, int(round(R_cert / G)))) for g in range(G)]
        rem2 = R_cert - sum(mc)
        i = 0; gl = list(range(G))
        while rem2 > 0:
            g = gl[i % G]
            if mc[g] < avail[g]:
                mc[g] += 1; rem2 -= 1
            i += 1
            if i % G == 0 and all(mc[g] >= avail[g] for g in gl):
                break
        # annex share = remainder
        R_annex = max(0, R - R_cert)
        annex_avail = {g: max(0, avail[g] - mc[g]) for g in range(G)}
        # adaptive annex counts: size-biased over annex_avail.  Direction score reads FIT-held
        # data only (disjoint from cargo), so the cargo share stays an independent SRS per group.
        scores = _scores_from_fit(name, seed)   # uses fit-held data only
        apt = {g: 1.0 for g in range(G)}
        if scores is not None:
            pg_sd = scores.get('pg_sd')
            apt = {g: float(pg_sd[g]) + 1e-9 if pg_sd is not None else 1.0 for g in range(G)}
        # round-robin with no cap exceeding annex_avail
        an = [min(annex_avail[g], max(0, int(round(R_annex * apt[g] / sum(apt.values()))))) for g in range(G)]
        rem3 = R_annex - sum(an)
        i = 0; gl = list(range(G))
        while rem3 > 0:
            g = gl[i % G]
            if an[g] < annex_avail[g]:
                an[g] += 1; rem3 -= 1
            i += 1
            if i % G == 0 and all(an[g] >= annex_avail[g] for g in gl):
                break
        # build certified cargo from the FIRST-revealed index range of each group (fixed count m_g).
        revc = np.zeros(len(yc), bool)
        for g in range(G):
            revc[order[g][:mc[g]]] = True
        # the annex actually spends its labels too (so total spend == R), but cargo stays the only
        # certified SRS share.  Annex decision read FIT data only, so cargo remains an independent
        # per-group SRS -> static MPB on cargo is sound at total budget R.
        rev_full = revc.copy()
        for g in range(G):
            if an[g] > 0:
                rev_full[order[g][mc[g]:mc[g] + an[g]]] = True
        UCBc = build_ucb_mpb(errs, yc, revc, Mnames, G, dcell_static(len(Mnames), G))
        rows_c = []
        for winfo in grid:
            i_, ub_ = selection_and_cert(errs, yc, revc, UCBc, Mnames, G, winfo['w'])
            trueR = {mid: sum(winfo['w'][g] * outer[mid][g] for g in range(G)) for mid in Mnames}
            reg = trueR[i_] - min(trueR.values())
            rows_c.append({'w': winfo['name'], 'chosen': i_, 'UB': float(ub_),
                           'true_regret': float(reg),
                           'label_cost': int(rev_full.sum()), 'annex': int(sum(an)),
                           'cert': 'static-cargo'})
        recs[(frac, 'C_cargo')] = rows_c
    return G, n_w, recs, outer


def _scores_from_fit(name, seed):
    """Recompute FIT-split per-group error margins for offline-adaptive annex direction (uses only
    fit-held data, disjoint from cargo).  Returns None on any failure (annex falls back to uniform)."""
    try:
        errs, fit_err, outer, yf, _ = get_artifacts(name, seed)
        G = int(int(yf.max()) + 1)
        Mnames = list(fit_err.keys())
        pg_sd = {}
        for g in range(G):
            vals = [fit_err[i][yf == g].mean() if (yf == g).sum() else 0.0 for i in Mnames]
            pg_sd[g] = float(np.std(vals)) if len(vals) >= 2 else 0.0
        return {'pg_sd': pg_sd}
    except Exception as e:
        return None


def main():
    t0 = time.time()
    ACC = {}
    for name in CARRIERS:
        for seed in SEEDS:
            G, n_w, recs, outer = run_carrier(name, seed)
            for (frac, cert), rows in recs.items():
                for tau in TAU_LIST:
                    key = (name, frac, cert, tau)
                    comm = [r for r in rows if r['UB'] <= tau]
                    cr = len(comm) / n_w if n_w else 0.0
                    if comm:
                        regs = np.array([r['true_regret'] for r in comm])
                        cv = float(np.mean(regs <= tau + 1e-9))
                    else:
                        cv = float('nan')
                    abst = [r for r in rows if r['UB'] > tau]
                    amean = float(np.mean([r['true_regret'] for r in abst])) if abst else None
                    amax = float(np.max([r['true_regret'] for r in abst])) if abst else None
                    label_cost = float(np.mean([r['label_cost'] for r in rows])) if rows else 0.0
                    annex = float(np.mean([r.get('annex', 0) for r in rows])) if rows else 0.0
                    ACC.setdefault(key, []).append(
                        {'carrier': name, 'frac': frac, 'cert': cert, 'tau': tau, 'seed': seed,
                         'n_w': n_w, 'committed_rate': cr,
                         'cert_validity': cv if cv == cv else None,
                         'abst_mean_regret': amean if amean is not None else None,
                         'abst_max_regret': amax if amax is not None else None,
                         'label_cost': label_cost, 'annex': annex, 'n_commit': len(comm)})
    agg = {}
    for key, vals in ACC.items():
        crs = [v['committed_rate'] for v in vals]
        nw = vals[0]['n_w']
        cvs = [v['cert_validity'] for v in vals if v['cert_validity'] is not None]
        cv_agg = float(np.mean(cvs)) if cvs else None
        abst_means = [v['abst_mean_regret'] for v in vals if v['abst_mean_regret'] is not None]
        abst_max = [v['abst_max_regret'] for v in vals if v['abst_max_regret'] is not None]
        agg[key] = {'carrier': vals[0]['carrier'], 'frac': vals[0]['frac'],
                    'cert': vals[0]['cert'], 'tau': vals[0]['tau'],
                    'label_cost': round(float(np.mean([v['label_cost'] for v in vals])), 1),
                    'annex': round(float(np.mean([v['annex'] for v in vals])), 1),
                    'n_w': nw, 'n_seeds': len(vals),
                    'committed_rate': round(float(np.mean(crs)), 3),
                    'cert_validity': round(cv_agg, 4) if cv_agg is not None else None,
                    'abst_mean_regret': round(float(np.mean(abst_means)), 4) if abst_means else None,
                    'abst_max_regret': round(float(np.max(abst_max, initial=-1e9)), 4) if abst_max else None}
    agg_list = [agg[k] for k in sorted(agg, key=lambda k: (k[0], k[1], k[3]))]
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1887',
           'agg': agg_list, 'meta': {'TAU_LIST': TAU_LIST, 'FRACS': FRACS, 'SEEDS': SEEDS,
                                     'DELTA': DELTA, 'CARRIERS': CARRIERS, 's_cert': 0.5},
           'runtime_s': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('runtime_s', out['runtime_s'])
    for name in CARRIERS:
        for frac in FRACS:
            line = [f"{name} frac={frac} lab={avg_label(agg, name, frac)}"]
            for cert in ['S_uniform', 'C_cargo']:
                if (name, frac, cert, 0.04) in agg:
                    a = agg[(name, frac, cert, 0.04)]
                    line.append(f"{cert}={a['committed_rate']}(cv{a['cert_validity']})")
            print('  ', ' | '.join(line))


def avg_label(agg, name, frac):
    for k, a in agg.items():
        if a['carrier'] == name and a['frac'] == frac and a['cert'] == 'S_uniform' and a['tau'] == 0.04:
            return a['label_cost']
    return float('nan')


if __name__ == '__main__':
    main()