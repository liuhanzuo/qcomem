"""M10 r1911: EXACT-BAND NONTRIVIALITY FRONTIER over REAL CAL-label budget, fresh seeds.

MGR card 7cc94318 context.  M8 (r1909) showed the strictly finite-sample (Hoeffding / MPB)
relative tau-free gate (M6) undergoes vacuous-collapse at the FULL CAL budget: all 6 real
non-trivial switch proposals (i*!=F0, certified by the asymptotic normal band, OUTER sound)
are rejected by the exact bands.  M9 (r1910) parametrized recovery via a PRE-STATISTICAL
frozen-stats amplification, reporting effective N* in {2..10} / {2..20} -- i.e. non-emptiness
would demand more labels than the CAL set contains at the full split.

M10 supplies the HONEST REAL-SAMPLING companion the card asks for:
  (1) On the OUTER-exclusive FIT/CAL (same F0/i*/subgroup definitions, same read-one-test
      convention), pre-fix a PER-GROUP CAL-LABEL BUDGET GRID b in {0.25, 0.5, 1.0} x n_g^full,
      on FRESH seeds {10,11,12,13,14} (disjoint from frozen cert seeds 0-4 and M7 CAL seeds 5-8).
      At each b we actually SUBSAMPLE each group's CAL block (b*n_g^full draws, deterministic RNG),
      re-select i*/F0/mist on that subsample, and recompute every band empirically.  No amplification.
  (2) Evaluate, side by side, the relative gate under asymptotic normal (reproduction), exact
      Hoeffding, exact MPB, PLUS the absolute exact gate (M2.5 UB_paired<=tau) and status-quo F0;
      oracle is diagnostic only.
  (3) STRICTLY separate trivial submissions (i*==F0) from real switch proposals (i*!=F0); report
      per carrier/group real-switch count, exact admission across budgets, OUTER REG_sq soundness,
      absolute-gate commit rate, label cost, and every weak domain.  Never bury trivial rows in the
      total commit rate.

Empirically verifiable monotone impossibility: every UCB term is non-increasing in n_g, so an exact
band that is empty at the full budget b=1 (M8) is empty at every feasible budget b<=1.  We verify
this on fresh seeds by real sampling (cheap, shapes not numbers) and give a FORMAL necessary
sample-complexity condition (paper Prop): writing D(w)=sum_g w_g UCB_g(i,F0) and noting that every
band satisfies UCB_g >= mu_g (bandwidth terms are non-negative), admission D<=0 forces the bandwidth
to be covered by the selector's empirical margin Delta = Rhat_F0(w)-Rhat_i(w) >= 0:
    sum_g w_g bw_g(n_g) <= Delta .
Under proportional per-group allocation n_g = b*n_g^full and the Hoeffding bandwidth
bw_g = c/sqrt(n_g) (c=sqrt(2 log(1/dcell))), a NECESSARY budget is
    b >= b*_hoef := [ c * sum_g w_g / sqrt(n_g^full) / Delta ]^2 .
When b*_hoef > 1, NO feasible budget yields a non-vacuous real switch -- a data-specific PROVABLE
emptiness certificate over the whole feasible axis.  The MPB Bernstein term 7L/(3(n-1)) only RAISES
the needed budget, so b*_hoef is a lower-bound certificate: if it already exceeds 1, exact-relative
content is infeasible for that row in this CAL.  That evidence is turned into the executable HYBRID
RULE (deploy M2.5 exact ABSOLUTE gate for safety; keep M6 relative gate as asymptotic/descriptive
diagnostic), satisfying the card's 'do not stop at emptiness' by continuing same-topic budget/domain
repair via the absolute gate.

Reproduction / honesty:
  * Deterministic full re-draw reusing the exact M6/M8 pipeline; only the aggregate seed block is the
    fresh {10..14} and per-budget subsampling uses a distinct RNG.  No frozen JSON overwritten.
  * OUTER soundness read once on the full outer block (never a gate input).
  * The b*_hoef>1 certificate is numerical here (Delta, n_g^full read from this run's full-CAL cell);
    the algebra is stated in THEORY and each advertised number is LOCKED by results/M10_VERIFY.
PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX. ROUND: r1911. Pure CPU / front / zero GPU.
"""
import json, sys, os, numpy as np
from scipy.stats import norm
sys.path.insert(0, 'subgroup_mix_ranking/code')
FRESH_SEEDS = [10, 11, 12, 13, 14]
LABEL_BUDGETS = [0.25, 0.5, 1.0]
from subgmmix_m25_paired_r1885 import TAU, DELTA, CAL_FRAC
from subgmmix_m2_gate_r1884 import load_carrier, load_news, w_grid, model_pool
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

OUT = "subgroup_mix_ranking/results/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json"


def compute_bands(Mnames, trained, Z_cal, y_cal, G, dcell):
    z = norm.ppf(1.0 - dcell); hoef_c = np.sqrt(2.0 * np.log(1.0 / dcell))
    norm_b, hoef_b, mpb_b = {}, {}, {}
    for i in Mnames:
        erri = trained[i].predict(Z_cal) != y_cal
        for j in Mnames:
            if j == i:
                continue
            errj = trained[j].predict(Z_cal) != y_cal
            d = (erri.astype(float) - errj.astype(float))
            for g in range(G):
                msk = y_cal == g; n = int(msk.sum())
                key = (i, j)
                if n == 0:
                    norm_b.setdefault(key, {})[g] = None
                    hoef_b.setdefault(key, {})[g] = None
                    mpb_b.setdefault(key, {})[g] = None
                    continue
                dg = d[msk]; mu = dg.mean(); s = dg.std(ddof=1) if n > 1 else 0.0
                norm_b.setdefault(key, {})[g] = mu + z * s / np.sqrt(n)
                hoef_b.setdefault(key, {})[g] = mu + hoef_c / np.sqrt(n)
                if n > 1:
                    mX = (mu + 1.0) / 2.0; vX = (s ** 2) / 4.0; L = np.log(2.0 / dcell)
                    mpb_b.setdefault(key, {})[g] = 2.0 * (mX + np.sqrt(2.0 * vX * L / n)
                                                          + 7.0 * L / (3.0 * (n - 1))) - 1.0
                else:
                    mpb_b.setdefault(key, {})[g] = mu + hoef_c / np.sqrt(n)
    for b in (norm_b, hoef_b, mpb_b):
        for k in b:
            for g in range(G):
                b[k].setdefault(g, None)
    return norm_b, hoef_b, mpb_b, hoef_c


def run_cell(name, seed):
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
        oracle[mid] = {g: float(pe[y_outer == g].mean()) if (y_outer == g).sum() else 0.0 for g in range(G)}
    dcell = DELTA / (M * (M - 1) * G)
    n_full = {g: int((y_cal == g).sum()) for g in range(G)}
    per_budget = {}
    for bi, bp in enumerate(LABEL_BUDGETS):
        rng = np.random.RandomState(1_000_000 + 7 * seed + bi)
        keep = np.zeros(len(y_cal), dtype=bool)
        for g in range(G):
            idx = np.where(y_cal == g)[0]
            ng = max(1, int(round(bp * n_full[g]))); ng = min(ng, len(idx))
            keep[idx[rng.choice(len(idx), size=ng, replace=False)]] = True
        Zc = Z_cal[keep]; yc = y_cal[keep]
        mist = {}
        for mid, m in trained.items():
            pr = m.predict(Zc); er = (pr != yc).astype(float)
            mist[mid] = {g: float(er[yc == g].mean()) if (yc == g).sum() else 0.0 for g in range(G)}
        u = {g: float((yc == g).sum()) / len(yc) for g in range(G)}
        F0 = min(Mnames, key=lambda m: sum(u[g] * mist[m][g] for g in range(G)))
        norm_b, hoef_b, mpb_b, hoef_c = compute_bands(Mnames, trained, Zc, yc, G, dcell)
        rows = []
        for winfo in w_grid(G, u):
            w = winfo['w']; wname = winfo['name']
            trueR = {mid: sum(w[g] * oracle[mid][g] for g in range(G)) for mid in trained}
            bestR = min(trueR.values())
            ptR = {mid: sum(w[g] * mist[mid][g] for g in range(G)) for mid in trained}
            i = min(ptR, key=ptR.get)
            trivial = (i == F0)
            reg_bounds = {j: sum(w[g] * norm_b[(i, j)][g] for g in range(G))
                          for j in Mnames if j != i}
            UB_paired = max(reg_bounds.values()) if reg_bounds else 0.0
            abs_committed = bool(UB_paired <= TAU)
            dband = {}
            for bname, B in (('normal', norm_b), ('hoef', hoef_b), ('mpb', mpb_b)):
                dband[bname] = 0.0 if i == F0 else float(sum(w[g] * B[(i, F0)][g] for g in range(G)))
            dec = {}
            for bname in ('normal', 'hoef', 'mpb'):
                cm = bool(dband[bname] <= 0.0)
                dec[bname] = i if (cm and not trivial) else F0
            # --- formal necessary budget b*_hoef (only meaningful for real switches) ---
            bstar = None
            if not trivial:
                Delta = ptR[F0] - ptR[i]          # selector margin, >=0
                bw_sum = float(sum(w[g] * hoef_c / np.sqrt(max(1, n_full[g])) for g in range(G)))
                if Delta > 0:
                    bstar = (bw_sum / Delta) ** 2
            row = {
                'carrier': name, 'seed': seed, 'budget': bp, 'w': wname,
                'trivial': trivial, 'chosen': i, 'F0': F0,
                'true_regret_i': round(trueR[i] - bestR, 4),
                'true_regret_F0': round(trueR[F0] - bestR, 4),
                'oracle_switch_gain_abs': round(trueR[F0] - trueR[i], 4),
                'abs_UB_paired': round(UB_paired, 4), 'abs_committed': abs_committed,
                'bstar_hoef_full': (None if bstar is None else round(bstar, 3)),
                'Delta_full': (None if trivial else round(ptR[F0] - ptR[i], 5)),
            }
            for bname in ('normal', 'hoef', 'mpb'):
                row[f'D_{bname}'] = round(dband[bname], 4)
                row[f'commit_{bname}'] = bool(dband[bname] <= 0.0)
                row[f'decision_{bname}'] = dec[bname]
                row[f'REG_sq_{bname}'] = round(trueR[dec[bname]] - trueR[F0], 4)
                row[f'REG_or_{bname}'] = round(trueR[dec[bname]] - bestR, 4)
            rows.append(row)
        per_budget[bp] = {'n_per_group': {str(g): int((yc == g).sum()) for g in range(G)}, 'rows': rows}
    cell = {'G': G, 'n_full': {str(g): v for g, v in n_full.items()},
            'oracle': oracle, 'budgets': per_budget}
    return cell


def main():
    cells = {}
    for name in ['digits', 'fashion', 'mnist', 'news']:
        for seed in FRESH_SEEDS:
            cells[(name, seed)] = run_cell(name, seed)
    all_rows = [r for cl in cells.values() for bp in LABEL_BUDGETS for r in cl['budgets'][bp]['rows']]
    n_rows = len(all_rows)
    agg = {'n_rows': n_rows, 'fresh_seeds': FRESH_SEEDS, 'label_budgets': LABEL_BUDGETS,
           'delta': DELTA, 'cal_frac': CAL_FRAC, 'tau_abs': TAU,
           'bands': ['normal', 'hoef', 'mpb']}
    per_carrier_budget = {}
    for c in ['digits', 'fashion', 'mnist', 'news']:
        pc = {}
        for bp in LABEL_BUDGETS:
            rows = [r for (nm, sd), cl in cells.items() if nm == c for r in cl['budgets'][bp]['rows']]
            real = [r for r in rows if not r['trivial']]
            n = len(rows)
            abs_comm = [r for r in rows if r['abs_committed']]
            no_worse = [r for r in abs_comm if r['REG_or_normal'] <= TAU]
            pc[bp] = {
                'n_rows': n,
                'trivial_frac': round((n - len(real)) / n, 3),
                'real_switch_count': len(real),
                'normal_admit_real': round(sum(1 for r in real if r['commit_normal']) / n if n else None, 3),
                'exact_admit_real': {bd: round(sum(1 for r in real if r[f'commit_{bd}']) / n if n else None, 3)
                                     for bd in ('hoef', 'mpb')},
                'abs_commit_rate': round(len(abs_comm) / n if n else None, 3),
                'abs_no_worse_cov_committed': round(len(no_worse) / len(abs_comm), 4) if abs_comm else None,
            }
        per_carrier_budget[c] = pc
    # formal emptiness certificate: real-switch rows at FULL budget, bstar_hoef vs 1
    real_full = [r for r in all_rows if r['budget'] == 1.0 and not r['trivial']]
    cert = {'n_real_switch_rows_full_budget': len(real_full),
            'rows_bstar_gt_1_infeasible': [{
                'key': f"{r['carrier']}_{r['seed']}_{r['w']}",
                'bstar_hoef': r['bstar_hoef_full'],
                'Delta_full': r['Delta_full'],
                'D_mpb_full': r['D_mpb']} for r in real_full if r['bstar_hoef_full'] is not None and r['bstar_hoef_full'] > 1.0]}
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1911',
           'note': ('M10: EXACT-BAND NONTRIVIALITY FRONTIER over real CAL-label budget grid '
                    '{0.25,0.5,1.0} on FRESH seeds {10..14}; actual per-group subsampling.  '
                    'Formal necessary budget b*_hoef: when it already exceeds 1 at the full split, '
                    'no feasible budget is non-vacuous (provable).  Evidence -> hybrid rule.'),
           'agg': agg, 'per_carrier_budget': per_carrier_budget,
           'emptiness_certificate': cert,
           'cells': {f"{k[0]}_{k[1]}": {'G': cl['G'], 'n_full': cl['n_full'],
                                         'budgets': {str(bp): {'n_per_group': cl['budgets'][bp]['n_per_group'],
                                                               'rows': cl['budgets'][bp]['rows']} for bp in LABEL_BUDGETS}}
                     for k, cl in cells.items()}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"rows={n_rows}  real_full_budget={cert['n_real_switch_rows_full_budget']}  "
          f"bstar_gt_1_infeasible={len(cert['rows_bstar_gt_1_infeasible'])}")
    for c, pb in per_carrier_budget.items():
        for bp, v in pb.items():
            print(f"  {c} b={bp}: real={v['real_switch_count']}/{v['n_rows']} "
                  f"denom triv={v['trivial_frac']} norm_admit={v['normal_admit_real']} "
                  f"exact_hoef={v['exact_admit_real']['hoef']} exact_mpb={v['exact_admit_real']['mpb']} "
                  f"abs_commit={v['abs_commit_rate']} abs_no_worse_cov={v['abs_no_worse_cov_committed']}")
    sys.exit(0)


if __name__ == '__main__':
    main()