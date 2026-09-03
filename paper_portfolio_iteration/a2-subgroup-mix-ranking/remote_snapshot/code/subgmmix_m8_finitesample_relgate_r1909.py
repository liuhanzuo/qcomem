"""M8 r1909: strictly finite-sample (non-asymptotic) bands in the tau-FREE relative gate.

The frozen conclusion (L689) leaves exactly one remaining item: "a strictly finite-sample
(non-asymptotic) tightened band."  The ABSOLUTE tau=0.04 gate already reports exact variants
(Hoeffding 0.097 / Maurer-Pontil 0.260 vs asymptotic normal 0.503, Tab.~table:frontier),
so that half is covered.  But the tau-free RELATIVE gate (M6, the promoted tau-elimination
branch in Sec.~app:tau) currently computes its gate statistic
    D(w) = sum_g w_g UCB[(i*,F0)][g] <= 0
with the ASYMPTOTIC normal paired-difference band only.  M6's no-worse-than-F0 soundness claim
therefore rests on asymptotic machinery, even though the manuscript presents M6 as a closed
tau-elimination.

M8 closes this: recompute the SAME relative gate under the two EXACT finite-sample bands that
the absolute pipeline already defines (recommended Hoeffding range-2 and Maurer-Pontil
empirical-Bernstein, both paired-difference one-sided), alongside the asymptotic normal for
reproduction.  Per row and per band-variant we record
    D_band(w) = sum_g w_g UCB_band[(i*,F0)][g],  m8_commit = D_band(w) <= 0,
    decision = i* if commit else F0,  REG_sq = R_decision(w) - R_F0(w) (<=0 = no-worse).
The CONTENT is the certificate-price of FINITE-SAMPLE RIGOR on the relative gate: because F0
(status-quo, collected-mixture best) is typically already near-optimal, the paired difference
i*-vs-F0 is small, so even the exact bands may admit most mixtures at no soundness loss.  If
exact-M8 upgrade-rate stays high and no-worse coverage stays 1.0, the relative gate is a place
where STRICT finite-sample bands are almost free --- a concrete, honest closure of the stated
remaining item (the manuscript's absolute-gate caveat is NOT the only cost locus).

Honesty / reproduction:
  * assert chosen/true_regret/UB_paired/committed/D_F0/m6_upgrade/decision/REG_sq EXACTLY
    equal the frozen r1906 file on the NORMAL band (same rows/splits/splits/trained models),
    so the new MPB/Hoeffding columns differentiate BAND RIGOR, not a re-implementation bug.
  * base fields chosen/true_regret/UB_paired/committed additionally asserted vs frozen
    SUBGMIX_M25_PAIRED_R1885_5SEED.json (same long-run_carrier path).
  * no-worse soundness verified ON OUTER (REG_sq<=0 fraction over committed rows), identical
    read-one-test convention to M6/M2.5.
  * exact finite-sample claims are statements about REG_sq vs F0, NOT absolute regret<=tau;
    absolute semantics remain the frozen M2.5 domain (disclosed, complementary).
PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX. ROUND: r1909. Pure CPU / front / zero GPU.
"""
import json, sys, numpy as np, os
from scipy.stats import norm
sys.path.insert(0, 'subgroup_mix_ranking/code')
SEEDS = [0, 1, 2, 3, 4]   # = frozen M6 extent (350 rows) — overrides r1885's 3-seed default
from subgmmix_m25_paired_r1885 import TAU, DELTA, CAL_FRAC
from subgmmix_m2_gate_r1884 import (load_carrier, load_news, w_grid, model_pool)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

OUT = "subgroup_mix_ranking/results/SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json"


def bands_paired(Mnames, trained, Z_cal, y_cal, G, dcell):
    """Return UCB[(i,j)][g] for the three one-sided paired-difference bands: normal, hoef, mpb.
    Mirrors the frozen absolute-pipeline formulas exactly (paired, d in [-1,1],
    X=(d+1)/2 in [0,1] for MPB)."""
    z = norm.ppf(1.0 - dcell)
    hoef = np.sqrt(2.0 * np.log(1.0 / dcell))
    norm_b, hoef_b, mpb_b = {}, {}, {}
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
                key = (i, j)
                if n == 0:
                    norm_b.setdefault(key, {})[g] = None
                    hoef_b.setdefault(key, {})[g] = None
                    mpb_b.setdefault(key, {})[g] = None
                    continue
                dg = d[msk]; mu = dg.mean(); s = dg.std(ddof=1) if n > 1 else 0.0
                norm_b.setdefault(key, {})[g] = mu + z * s / np.sqrt(n)
                hoef_b.setdefault(key, {})[g] = mu + hoef / np.sqrt(n)
                if n > 1:
                    mX = (mu + 1.0) / 2.0
                    vX = (s ** 2) / 4.0
                    L = np.log(2.0 / dcell)
                    UCBX = mX + np.sqrt(2.0 * vX * L / n) + 7.0 * L / (3.0 * (n - 1))
                    mpb_b.setdefault(key, {})[g] = 2.0 * UCBX - 1.0
                else:
                    mpb_b.setdefault(key, {})[g] = mu + hoef / np.sqrt(n)
    for b in (norm_b, hoef_b, mpb_b):
        for k in b:
            if len(b[k]) < G:
                for g in range(G):
                    b[k].setdefault(g, None)
    return norm_b, hoef_b, mpb_b


def run_m8(name, seed):
    X, y, kind = (load_news() if name == 'news' else load_carrier(name))
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
    dcell = DELTA / (M * (M - 1) * G)
    norm_b, hoef_b, mpb_b = bands_paired(Mnames, trained, Z_cal, y_cal, G, dcell)
    u = {g: (y_cal == g).sum() / len(y_cal) for g in range(G)}
    F0 = min(Mnames, key=lambda m: sum(u[g] * mist[m][g] for g in range(G)))
    rows = []
    for winfo in w_grid(G, u):
        w = winfo['w']; wname = winfo['name']
        trueR = {mid: sum(w[g] * oracle[mid][g] for g in range(G)) for mid in trained}
        bestR = min(trueR.values())
        ptR = {mid: sum(w[g] * mist[mid][g] for g in range(G)) for mid in trained}
        i = min(ptR, key=ptR.get)   # i*(w), same selector as frozen M2.5
        # absolute-band fields (reproduction vs frozen files)
        reg_bounds = {}
        for j in Mnames:
            if j == i: continue
            reg_bounds[j] = sum(w[g] * norm_b[(i, j)][g] for g in range(G))
        UB_paired = max(reg_bounds.values()) if reg_bounds else 0.0
        committed_tau = bool(UB_paired <= TAU)
        # relative-gate per band variant
        dband = {}
        raw = {'normal': norm_b, 'hoef': hoef_b, 'mpb': mpb_b}
        for bname, B in raw.items():
            if i == F0:
                dband[bname] = 0.0
            else:
                dband[bname] = sum(w[g] * B[(i, F0)][g] for g in range(G))
        dec = {}
        for bname in raw:
            commit = bool(dband[bname] <= 0.0)
            dec[bname] = i if commit else F0
        row = {
            'carrier': name, 'seed': seed, 'w': wname,
            'chosen': i, 'F0': F0,
            'true_regret': round(trueR[i] - bestR, 4),
            'UB_paired': round(UB_paired, 4),
            'committed': committed_tau,
            'true_regret_F0': round(trueR[F0] - bestR, 4),
            'D_normal': round(dband['normal'], 4),
            'D_hoef': round(dband['hoef'], 4),
            'D_mpb': round(dband['mpb'], 4),
        }
        for bname in raw:
            decmodel = dec[bname]
            row[f'decision_{bname}'] = decmodel
            row[f'commit_{bname}'] = bool(dband[bname] <= 0.0)
            row[f'REG_sq_{bname}'] = round(trueR[decmodel] - trueR[F0], 4)
            row[f'REG_or_{bname}'] = round(trueR[decmodel] - bestR, 4)
        rows.append(row)
    return rows, {'oracle': oracle, 'mist': mist}


def main():
    all_rows = []
    for name in ['digits', 'fashion', 'mnist', 'news']:
        for seed in SEEDS:
            rows, _ = run_m8(name, seed)
            all_rows.extend(rows)
    # ---- reproduction assert #1 vs frozen M6 file (NORMAL relative gate = r1906) ----
    m6 = json.load(open("subgroup_mix_ranking/results/SUBGMIX_M6_UPGRADEGATE_R1906.json"))
    m6_rows = {(r['carrier'], r['seed'], r['w']): r for r in m6['rows']}
    mism_m6 = []
    for r in all_rows:
        fr = m6_rows.get((r['carrier'], r['seed'], r['w']))
        if fr is None:
            mism_m6.append((r['carrier'], r['seed'], r['w'], 'MISSING_IN_M6'))
            continue
        # m6_upgrade <-> commit_i* (== commit_normal), decision == decision_normal
        if fr['m6_upgrade'] != r['commit_normal']:
            mism_m6.append((r['carrier'], r['seed'], r['w'], 'm6_upgrade-vs-commit_normal', fr['m6_upgrade'], r['commit_normal']))
        if fr['decision'] != r['decision_normal']:
            mism_m6.append((r['carrier'], r['seed'], r['w'], 'decision-vs-decision_normal', fr['decision'], r['decision_normal']))
        if abs(fr['D_F0'] - r['D_normal']) > 1e-9:
            mism_m6.append((r['carrier'], r['seed'], r['w'], 'D_F0-vs-D_normal', fr['D_F0'], r['D_normal']))
        if abs(fr['REG_sq'] - r['REG_sq_normal']) > 1e-9:
            mism_m6.append((r['carrier'], r['seed'], r['w'], 'REG_sq-vs-REG_sq_normal', fr['REG_sq'], r['REG_sq_normal']))
        if abs(fr['REG_or'] - r['REG_or_normal']) > 1e-9:
            mism_m6.append((r['carrier'], r['seed'], r['w'], 'REG_or-vs-REG_or_normal', fr['REG_or'], r['REG_or_normal']))
        if fr['chosen'] != r['chosen'] or fr['F0'] != r['F0']:
            mism_m6.append((r['carrier'], r['seed'], r['w'], 'chosen/F0'))
    # ---- reproduction assert #2 vs frozen 5-seed M2.5 file (absolute base fields) ----
    frozen = json.load(open("subgroup_mix_ranking/results/SUBGMIX_M25_PAIRED_R1885_5SEED.json"))
    frozen_rows = {(r['carrier'], r['seed'], r['w']): r for r in frozen['rows']}
    mism_fz = []
    for r in all_rows:
        fr = frozen_rows.get((r['carrier'], r['seed'], r['w']))
        if fr is None:
            mism_fz.append((r['carrier'], r['seed'], r['w'], 'MISSING_IN_FROZEN')); continue
        for k in ['chosen', 'true_regret', 'UB_paired', 'committed']:
            if fr[k] != r[k]:
                mism_fz.append((r['carrier'], r['seed'], r['w'], k, fr[k], r[k]))
    assert not mism_m6, f"M6 reproduction FAILED ({len(mism_m6)}): {mism_m6[:5]}"
    assert not mism_fz, f"M2.5 reproduction FAILED ({len(mism_fz)}): {mism_fz[:5]}"

    # ---- M8 aggregate: per band variant ----
    bands = ['normal', 'hoef', 'mpb']
    agg = {'n_rows': len(all_rows), 'seeds': SEEDS, 'delta': DELTA,
           'cal_frac': CAL_FRAC, 'tau_abs': TAU,
           'finite_sample_bands': ['hoef', 'mpb'], 'asymptotic_band': 'normal'}
    for b in bands:
        commit = [r for r in all_rows if r[f'commit_{b}']]
        rate = len(commit) / len(all_rows)
        rsq = [r[f'REG_sq_{b}'] for r in commit] if commit else []
        agg[b] = {
            'commit_rate': round(rate, 4),
            'sq_no_worse_cov_upgraded': round(float(np.mean([x <= 1e-9 for x in rsq])), 4) if rsq else None,
            'sq_mean_upgraded': round(float(np.mean(rsq)), 4) if rsq else None,
            'sq_max_upgraded': round(float(np.max(rsq)), 4) if rsq else None,
            'or_mean_all': round(float(np.mean([r[f'REG_or_{b}'] for r in all_rows])), 4),
            'or_max_all': round(float(np.max([r[f'REG_or_{b}'] for r in all_rows])), 4),
            'or_max_committed': round(float(np.max(rsq and [r[f'REG_or_{b}'] for r in commit] or [0.0])), 4) if commit else None,
        }
    # per-carrier (mpb-focused: vector-heterogeneity)
    per_carrier = {}
    for c in ['digits', 'fashion', 'mnist', 'news']:
        cr = [r for r in all_rows if r['carrier'] == c]
        pc = {}
        for b in bands:
            cc = [r for r in cr if r[f'commit_{b}']]
            rsq = [r[f'REG_sq_{b}'] for r in cc] if cc else []
            pc[b] = {'commit_rate': round(len(cc) / len(cr), 3),
                     'sq_no_worse_cov_upgraded': round(float(np.mean([x <= 1e-9 for x in rsq])), 4) if rsq else None}
        pc['F0_is_chosen_frac'] = round(np.mean([r['chosen'] == r['F0'] for r in cr]), 3)
        per_carrier[c] = pc
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1909',
           'note': ('M8: strictly finite-sample (Hoeffding/Maurer-Pontil) paired-difference bands '
                    'in the tau-FREE relative gate; asymptotic normal kept for reproduction. '
                    'Closes the Conclusion L689 "strictly finite-sample tightened band" item for the '
                    'relative gate. base fields/decision/REG_sq asserted EXACT vs frozen r1906+r1903 files.'),
           'agg': agg, 'per_carrier': per_carrier, 'rows': all_rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print("M6 reproduction: EXACT (normal band = frozen r1906).  M2.5 base: EXACT (frozen 5seed).")
    print("AGG:", json.dumps(agg, indent=2))
    for c, v in per_carrier.items():
        print(f"  {c}: {v}")
    sys.exit(0)


if __name__ == '__main__':
    main()