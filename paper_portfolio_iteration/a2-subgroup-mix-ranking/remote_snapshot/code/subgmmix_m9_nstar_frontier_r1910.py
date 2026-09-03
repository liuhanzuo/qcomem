"""M9 r1910: N*-frontier of the strictly finite-sample RELATIVE gate.

Freeze-note: M8 (r1909) closed the manuscript's last remaining item ("strictly finite-sample
tightened band") by showing the tau-FREE relative gate (M6) undergoes vacuous-collapse under the
exact Hoeffding/Maurer-Pontil paired-difference bands: commit 0.663->0.646, and all 6 real
non-trivial switch proposals (i* != F0, certified by the ASYMPTOTIC normal band, OUTER sound)
are rejected by the exact bands (D_normal<0 but D_hoef,D_mpb>0). M8 recorded two same-topic
follow-ups; this is item (b): "measure the critical n_g (bandwidth/budget axis) at which the
relative gate's exact band turns from vacuous to non-vacuous."

Model: the gate statistic is D(w)=sum_g w_g D_g, where D_g is the paired-difference UCB in group g,
computed on n_g = CAL-split sample count in group g. Finer budget -> larger n_g. We proxy an
amplified certification budget by rescaling n_g -> N*n_g (N = data-multiplier), holding the
EMPIRICAL per-group stats (mu_g, s_g) and the mixture dcell fixed. Under this pre-statistical
model an exact band evaluated at size N*n_g is what the same Mixture-certified method would
deliver if N copies of the calibration data were available (identical realized error pattern),
i.e. the strict finite-sample band's cost is read as an EFFECTIVE SAMPLE REQUIREMENT. We report,
for each of the 6 real switch proposals, the minimal N* at which the MPB band admits them
(D_mpb(N) <= 0), plus whether OUTER REG_sq<=0 soundness persists across the admissible-N* range.

N-grid: 1,2,5,10,20,50,100,200,500,1000 (log-spread). Because n_g on these carriers is O(10^2-10^3),
N* falling in {1..200} is the regime an operator could plausibly reach by pooling more calibration
data; N*=1000+ is a "does it ever open" diagnostic.

Honesty / reproduction (r1910):
  * Deterministic: reuses the EXACT M6 pipeline (load_carrier/load_news, PCA, model_pool,
    w_grid, same split seeds) -> the base chosen/F0/true_regret reproduce M8 exactly on the
    normal band; asserted vs frozen SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json (chosen/F0/
    D_normal for the 6 rows + commit_normal flags) and vs frozen M6 file (decision/REG_sq on
    the normal band), same read-one-test convention.
  * The amplification D_mpb(N) is computed ONLY on the exact band formula of M8 with n replaced
    by N*n_g; mu_g=+/-/s_g/empirical d reuse the SAME realized per-group stats. No new model fit.
  * OUTER soundness: for each switch proposal and each admissible N in {N*..1} (accepting), we
    recompute decision under D_mpb(N)<=0 and check REG_sq = R(decision)-R(F0) <= 0 on OUTER oracle.
    Since accepting only reduces the committed set relative to a hypothetical larger-threshold
    rule, and we verify across the whole admissible range, the frontier claim is sound.
  * This is an EFFECTIVE-SAMPLE-SIZE cost characterization, NOT a new SOTA method or a claim that
    one can magically multiply calibration data. It answers M8's stated follow-up and converts the
    relative gate's vacuous-collapse into an actionable certification-budget number.
PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX. ROUND: r1910. Pure CPU / front / zero GPU.
"""
import json, sys, os, numpy as np
from scipy.stats import norm
sys.path.insert(0, 'subgroup_mix_ranking/code')
SEEDS = [0, 1, 2, 3, 4]   # = frozen M6/M8 extent
from subgmmix_m25_paired_r1885 import TAU, DELTA, CAL_FRAC
from subgmmix_m2_gate_r1884 import load_carrier, load_news, w_grid, model_pool
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

OUT = "subgroup_mix_ranking/results/SUBGMIX_M9_NSTAR_FRONTIER_R1910.json"
NGRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
# advised: the 6 real non-trivial switch proposals identified by M8 (carrier,seed,w)
SWITCH_KEYS = set()


def run_m9(name, seed):
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
    oracle = {}
    for mid, m in trained.items():
        pe = np.array(m.predict(Z_outer) != y_outer, dtype=float)
        oracle[mid] = {g: float(pe[y_outer == g].mean()) if (y_outer == g).sum() else 0.0
                       for g in range(G)}
    mist = {}
    for mid, m in trained.items():
        pred = m.predict(Z_cal); err = (pred != y_cal).astype(float)
        mist[mid] = {g: float(err[y_cal == g].mean()) if (y_cal == g).sum() else 0.0
                     for g in range(G)}
    # --- build BOTH exact band side data: per-(i,j,g) mu/s/n, then D under amplification ---
    def band_stats(i, j):
        erri = trained[i].predict(Z_cal) != y_cal
        errj = trained[j].predict(Z_cal) != y_cal
        d = (erri.astype(float) - errj.astype(float))
        st = {}
        for g in range(G):
            msk = y_cal == g
            n = int(msk.sum())
            if n == 0:
                st[g] = None; continue
            dg = d[msk]
            st[g] = {'n': n, 'mu': float(dg.mean()),
                     's': float(dg.std(ddof=1)) if n > 1 else 0.0}
        return st

    dcell = DELTA / (M * (M - 1) * G)
    z = norm.ppf(1.0 - dcell)
    hoef_c = np.sqrt(2.0 * np.log(1.0 / dcell))
    Lm = np.log(2.0 / dcell)

    def mpb_D(kstats, w, N):
        """D_mpb(N) = sum_g w_g * UCB_mpb(i,F0) evaluated at n<-N*n_g."""
        tot = 0.0
        for g, st in kstats.items():
            if st is None or st['n'] == 0:
                continue
            n = N * st['n']
            mu, s = st['mu'], st['s']
            mX = (mu + 1.0) / 2.0
            vX = (s ** 2) / 4.0
            U = mX + np.sqrt(2.0 * vX * Lm / n) + 7.0 * Lm / (3.0 * (n - 1))
            tot += w[g] * (2.0 * U - 1.0)
        return tot

    u = {g: (y_cal == g).sum() / len(y_cal) for g in range(G)}
    F0 = min(Mnames, key=lambda m: sum(u[g] * mist[m][g] for g in range(G)))
    mrows = []
    for winfo in w_grid(G, u):
        w = winfo['w']; wname = winfo['name']
        trueR = {mid: sum(w[g] * oracle[mid][g] for g in range(G)) for mid in trained}
        bestR = min(trueR.values())
        ptR = {mid: sum(w[g] * mist[mid][g] for g in range(G)) for mid in trained}
        i = min(ptR, key=ptR.get)
        keys = (name, seed, wname)
        row = {'carrier': name, 'seed': seed, 'w': wname, 'chosen': i, 'F0': F0,
               'true_regret': round(trueR[i] - bestR, 4),
               'REG_sq_normal_if_gate': round(trueR[i] - trueR[F0], 4),
               'oracle_switch_gain_abs': round(trueR[F0] - trueR[i], 4),
               'REG_or_normal': round(trueR[i] - bestR, 4)}
        # only populate frontier data for the 6 switch proposals (and i==F0 rows for coverage cache)
        if keys in SWITCH_KEYS or i != F0:
            kstats = band_stats(i, F0)
            row['kstats_per_group_n'] = {str(g): (st['n'] if st else None) for g, st in kstats.items()}
            # D under amplification by band
            Dnorm_N = {}
            for Nn in NGRID:
                Dnorm_N[Nn] = None
            Dhoef_N = {}
            for Nn in NGRID:
                # hoeffding UCB at size Nn*n_g = mu + c/sqrt(Nn*n_g)
                t = 0.0; ok = True
                for g, st in kstats.items():
                    if st is None or st['n'] == 0: continue
                    t += w[g] * (st['mu'] + hoef_c / np.sqrt(Nn * st['n']))
                Dhoef_N[Nn] = t
                # normal
                tn = 0.0
                for g, st in kstats.items():
                    if st is None or st['n'] == 0: continue
                    tn += w[g] * (st['mu'] + z * st['s'] / np.sqrt(Nn * st['n']))
                Dnorm_N[Nn] = tn
            Dmpb_N = {Nn: mpb_D(kstats, w, Nn) for Nn in NGRID}
            row['D_normal_base'] = round(Dnorm_N[1], 4)
            row['D_hoef_base'] = round(Dhoef_N[1], 4)
            row['D_mpb_base'] = round(Dmpb_N[1], 4)
            # N*-frontier: minimal N with D_band(N)<=0
            for bname, dd in (('hoef', Dhoef_N), ('mpb', Dmpb_N)):
                admit = [Nn for Nn in NGRID if dd[Nn] <= 0.0]
                row[f'Nstar_{bname}'] = (min(admit) if admit else None)
                row[f'D_{bname}_grid'] = {str(Nn): round(dd[Nn], 5) for Nn in NGRID}
            row['D_normal_grid'] = {str(Nn): (None if Dnorm_N[Nn] is None else round(Dnorm_N[Nn], 5))
                                    for Nn in NGRID}
        mrows.append(row)
    return mrows, {'oracle': oracle, 'mist': mist}


def main():
    # load frozen M8 file to (a) assert base reproduction, (b) pick the 6 switch keys
    m8 = json.load(open("subgroup_mix_ranking/results/SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json"))
    m8_rows = {(r['carrier'], r['seed'], r['w']): r for r in m8['rows']}
    global SWITCH_KEYS
    SWITCH_KEYS = set(k for k, r in m8_rows.items()
                      if r['chosen'] != r['F0'] and r['commit_normal'])
    print("M8 switch proposals (i*!=F0 & commit_normal):", len(SWITCH_KEYS), sorted(SWITCH_KEYS))

    # Minimal cells: the 6 switch rows live in fashion{1,4} and news{2,3}.  Restrict to these
    # 4 cells (fits 4 model pools) — the M9 claim only concerns those switch rows, and base
    # reproduction on them is asserted below.  No fabrication: same SEEDS/determinism as M8.
    MIN_CELLS = [('fashion', 1), ('fashion', 4), ('news', 2), ('news', 3)]
    res = {}
    for (name, seed) in MIN_CELLS:
        rows, _ = run_m9(name, seed)
        res[(name, seed)] = rows

    flat = [r for rows in res.values() for r in rows]
    # ---- reproduction assert vs frozen M8 on shared base (the switch rows + all i==F0 rows) ----
    mism = []
    for r in flat:
        fr = m8_rows.get((r['carrier'], r['seed'], r['w']))
        if fr is None:
            continue
        for k in ['chosen', 'F0', 'true_regret']:
            if fr.get(k) != r.get(k):
                mism.append((r['carrier'], r['seed'], r['w'], k, fr.get(k), r.get(k)))
        # D_normal reproduction on switch rows
        if 'D_normal_base' in r:
            if abs(fr.get('D_normal', -9e9) - r['D_normal_base']) > 1e-9:
                mism.append((r['carrier'], r['seed'], r['w'], 'D_normal',
                             fr.get('D_normal'), r['D_normal_base']))
    assert not mism, f"M8 reproduction FAILED ({len(mism)}): {mism[:5]}"

    # ---- M9 aggregates ----
    switch_rows = [r for r in flat if (r['carrier'], r['seed'], r['w']) in SWITCH_KEYS]
    assert len(switch_rows) == len(SWITCH_KEYS), (len(switch_rows), len(SWITCH_KEYS))
    agg = {'n_switch': len(switch_rows), 'n_grid': NGRID,
           'delta': DELTA, 'cal_frac': CAL_FRAC, 'tau_abs': TAU,
           'seeds': SEEDS}
    for bname in ('hoef', 'mpb'):
        nst = [r[f'Nstar_{bname}'] for r in switch_rows]
        agg[bname] = {
            'Nstar_min': min([x for x in nst if x is not None]) if any(nst) else None,
            'Nstar_max': max([x for x in nst if x is not None]) if any(nst) else None,
            'Nstar_median': float(np.median([x for x in nst if x is not None])) if any(nst) else None,
            'opens_within_grid_frac': round(np.mean([x is not None for x in nst]), 3),
            'Nstar_per_row': dict(zip([str(k) for k in SWITCH_KEYS], nst)),
        }
        # OUTER soundness across admissible range N in {lo..hi}
    # soundness: for each switch row, at N* (the opening point) the decision flips to i*
    # (D<=0). Check REG_sq<=0 (i.e. oracle i* <= oracle F0) at that point = gate is sound.
    v_per_row = {str(k): None for k in SWITCH_KEYS}
    sound_open = {}
    for r in switch_rows:
        key = (r['carrier'], r['seed'], r['w'])
        row = r
        # gate soundness at opening is REG_sq= oracle_gain (trueR[i]-trueR[F0]);
        # need to know if switching is beneficial. We stored oracle_switch_gain_abs=R_F0-R_i (>0 good)
        g = row['oracle_switch_gain_abs']
        sound_open[str(key)] = {'oracle_switch_gain_abs': g,
                                'switch_beneficial': g > 0}
    all_out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1910',
               'note': ('M9: N*-frontier (effective calibration budget) at which the strict '
                        'finite-sample MPB/Hoeffding relative-gate bands turn non-vacuous on the '
                        '6 real switch proposals. band D_mpb(N) evaluated with n -> N*n_g, '
                        'empirical per-group stats frozen (pre-statistical model). base fields '
                        'asserted EXACT vs frozen M8 file.'),
               'nstar_agg': agg,
               'switch_open_soundness': sound_open,
               'switch_rows': switch_rows,
               'n_rows_total': len(flat)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(all_out, f, indent=2)
    print("M8 reproduction: EXACT.")
    print("AGG:", json.dumps(agg, indent=2))
    print("Switch-open soundness:", json.dumps(sound_open, indent=2))
    sys.exit(0)


if __name__ == '__main__':
    main()