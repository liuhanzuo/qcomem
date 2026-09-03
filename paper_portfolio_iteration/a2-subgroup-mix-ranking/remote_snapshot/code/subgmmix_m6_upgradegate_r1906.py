"""M6 r1906: tau-FREE safe-upgrade gate for subgroup-mix ranking.

M2/M2.5 (frozen, r1884/r1885) commit i*(w) iff its paired regret-vs-ORACLE-BEST
is certified <= operator-set tau.  The conclusion of the frozen paper names
"a tau-selection protocol" as the open follow-on: tau is an operator knob and
"abstain" currently has no clean deployed object, so a deployer cannot reason
about whether the framework is worth turning on.

M6 replaces the tau knob by a deployable *safe upgrade* target.  Let
    F0 = argmin_i sum_g u_g p_hat_{i,g}      (collected-mixture point pick
                                              = the status-quo single-point
                                              selector this framework critiques)
be what the operator would run WITHOUT the framework.  At a deployed mixture w,
M6's decision is:
    commit i*(w)  iff  sum_g w_g UCB[(i*,F0)][g] <= 0
    else keep F0                          (honest fallback = run the status quo)
The gate's soundness is a *paired-difference* claim (same points, same group,
shared error cancels, one-sided normal UCB from the frozen M2.5 pipeline):
    D(w) = sum_g w_g UCB[(i*,F0)][g] <= 0  ==>  R_{i*}(w) <= R_{F0}(w)
    with joint coverage >= 1-delta over all ordered pairs x groups (Bonferroni),
    so conditioning on the CAL-selected i* and F0 is safe (simultaneous bands).
On the joint event the committed choice is certified NO-WORSE than the status-quo
F0 at THIS w --- no tau anywhere.

Honest endpoints (OUTER-settled):
  decision(w) = i* if committed else F0.
  REG_sq(w)  = R_decision(w) - R_F0(w)   regret vs status quo; <=0 = framework
                                          helped or tied.  Trivially <=0 on
                                          abstain (decision=F0); the content is
                                          the committed-rows REG_sq<=0 k proof.
  REG_or(w)  = R_decision(w) - min_j R_j(w)   standard regret vs oracle-best.
  upgrade_rate            fraction of w where the framework switches to i*.
  sq_no_worse_cov         over committed rows, fraction with REG_sq <= 0 (OUTER).
  or_max_committed        max REG_or over committed rows (bounded by both i* and
                          F0's reg; the tau-free claim is only vs F0, disclosed).

Relationship to the frozen M2.5 tau=0.04 gate: same rows/splits/bands, so the
base fields (chosen, true_regret, UB_paired, committed at tau=0.04) must EXACTLY
reproduce SUBGMIX_M25_PAIRED_R1885_5SEED.json.  That reproduction is asserted
before M6 numbers are trusted.

PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX. ROUND r1906. Pure CPU / front / zero
GPU.  One read of the test set only (= the OUTER oracle-regret oracle, identical
to the frozen pipeline).
"""
import json, sys, numpy as np
from scipy.stats import norm
sys.path.insert(0, 'subgroup_mix_ranking/code')
from subgmmix_m25_paired_r1885 import run_carrier, TAU, DELTA, CAL_FRAC, SEEDS
from subgmmix_m2_gate_r1884 import (load_carrier, load_news, w_grid, model_pool)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

OUT = "subgroup_mix_ranking/results/SUBGMIX_M6_UPGRADEGATE_R1906.json"


def run_m6(name, seed):
    """Mirror run_carrier's exact split/band logic but also return F0 and the
    full per-(i,j,g) normal UCB matrix needed for the F0-vs-everything gate."""
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
    # oracle (OUTER, diagnostics only)
    oracle = {}
    for mid, m in trained.items():
        pe = np.array(m.predict(Z_outer) != y_outer, dtype=float)
        oracle[mid] = {g: float(pe[y_outer == g].mean()) if (y_outer == g).sum() else 0.0
                       for g in range(G)}
    # per-group point estimates (CAL), for selection (i*) and status-quo (F0)
    mist = {}
    for mid, m in trained.items():
        pred = m.predict(Z_cal); err = (pred != y_cal).astype(float)
        mist[mid] = {g: float(err[y_cal == g].mean()) if (y_cal == g).sum() else 0.0
                     for g in range(G)}
    # normal one-sided paired-difference UCB, same Bonferroni split as frozen
    dcell = DELTA / (M * (M - 1) * G)
    z = norm.ppf(1.0 - dcell)
    UCB = {}   # UCB[(i,j)][g]
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
    # status-quo: single-point best under collected mixture u
    F0 = min(Mnames, key=lambda m: sum(u[g] * mist[m][g] for g in range(G)))
    rows = []
    for winfo in w_grid(G, u):
        w = winfo['w']; wname = winfo['name']
        trueR = {mid: sum(w[g] * oracle[mid][g] for g in range(G)) for mid in trained}
        bestR = min(trueR.values())
        ptR = {mid: sum(w[g] * mist[mid][g] for g in range(G)) for mid in trained}
        i = min(ptR, key=ptR.get)   # i*(w), same selector as frozen M2.5
        # frozen M2.5 certificate fields (for reproduction assert + relation)
        reg_bounds = {}
        for j in Mnames:
            if j == i: continue
            reg_bounds[j] = sum(w[g] * UCB[(i, j)][g] for g in range(G))
        UB_paired = max(reg_bounds.values()) if reg_bounds else 0.0
        committed_tau = bool(UB_paired <= TAU)      # frozen M2.5 gate
        # M6 tau-free upgrade gate: i* no-worse than F0 at w
        if i == F0:
            D = 0.0
        else:
            D = sum(w[g] * UCB[(i, F0)][g] for g in range(G))
        m6_commit = bool(D <= 0.0)
        decision = i if m6_commit else F0
        R_sq = trueR[decision] - trueR[F0]          # regret vs status-quo
        R_or = trueR[decision] - bestR              # standard regret vs best
        true_regret_i = trueR[i] - bestR            # frozen field (regret if chose i*)
        rows.append({
            'carrier': name, 'seed': seed, 'w': wname,
            'chosen': i, 'F0': F0, 'decision': decision,
            'true_regret': round(true_regret_i, 4),   # frozen field (i* vs best)
            'UB_paired': round(UB_paired, 4),        # frozen field
            'committed': committed_tau,              # frozen M2.5 gate
            'm6_upgrade': m6_commit,                 # M6 tau-free switch-to-i*
            'D_F0': round(D, 4),                     # M6 gate stat
            'REG_sq': round(R_sq, 4),                # vs status-quo
            'REG_or': round(R_or, 4),                # vs oracle-best
        })
    return rows, {'F0': F0, 'oracle': oracle, 'mist': mist}


def main():
    all_rows = []
    extra = {}
    for name in ['digits', 'fashion', 'mnist', 'news']:
        for seed in SEEDS:
            rows, ex = run_m6(name, seed)
            all_rows.extend(rows)
            extra[(name, seed)] = ex
    # ---- reproduction assert vs frozen 5-seed file ----
    frozen = json.load(open("subgroup_mix_ranking/results/SUBGMIX_M25_PAIRED_R1885_5SEED.json"))
    frozen_rows = {(r['carrier'], r['seed'], r['w']): r for r in frozen['rows']}
    mism = []
    for r in all_rows:
        fr = frozen_rows.get((r['carrier'], r['seed'], r['w']))
        if fr is None:
            mism.append((r['carrier'], r['seed'], r['w'], 'MISSING_IN_FROZEN'))
            continue
        for k in ['chosen', 'true_regret', 'UB_paired', 'committed']:
            if fr[k] != r[k]:
                mism.append((r['carrier'], r['seed'], r['w'], k, fr[k], r[k]))
    assert not mism, f"M2.5 reproduction FAILED on {len(mism)} fields: {mism[:5]}"
    # ---- M6 aggregate ----
    agg = {'n_rows': len(all_rows), 'seeds': SEEDS, 'tau_free': True,
           'delta': DELTA, 'cal_frac': CAL_FRAC}
    n_up = sum(1 for r in all_rows if r['m6_upgrade'])
    up_rows = [r for r in all_rows if r['m6_upgrade']]
    sq_all = [r['REG_sq'] for r in all_rows]
    agg['upgrade_rate'] = round(n_up / len(all_rows), 4)
    agg['sq_no_worse_cov_all'] = round(float(np.mean([x <= 1e-9 for x in sq_all])), 4)
    agg['sq_mean_all'] = round(float(np.mean(sq_all)), 4)
    agg['sq_max_all'] = round(float(np.max(sq_all)), 4)
    agg['sq_mean_upgraded'] = round(float(np.mean([r['REG_sq'] for r in up_rows])), 4) if up_rows else None
    agg['sq_max_upgraded'] = round(float(np.max([r['REG_sq'] for r in up_rows])), 4) if up_rows else None
    agg['sq_no_worse_cov_upgraded'] = round(float(np.mean(
        [x['REG_sq'] <= 1e-9 for x in up_rows])), 4) if up_rows else None
    agg['or_max_all'] = round(float(np.max([r['REG_or'] for r in all_rows])), 4)
    agg['or_max_committed'] = round(float(np.max([r['REG_or'] for r in up_rows])), 4) if up_rows else None
    agg['or_mean_all'] = round(float(np.mean([r['REG_or'] for r in all_rows])), 4)
    # per-carrier
    per_carrier = {}
    for c in ['digits', 'fashion', 'mnist', 'news']:
        cr = [r for r in all_rows if r['carrier'] == c]
        cur = [r for r in cr if r['m6_upgrade']]
        per_carrier[c] = {
            'n': len(cr),
            'upgrade_rate': round(sum(1 for r in cr if r['m6_upgrade']) / len(cr), 4),
            'sq_no_worse_cov_all': round(float(np.mean([x['REG_sq'] <= 1e-9 for x in cr])), 4),
            'sq_mean_all': round(float(np.mean([r['REG_sq'] for r in cr])), 4),
            'sq_mean_upgraded': round(float(np.mean([r['REG_sq'] for r in cur])), 4) if cur else None,
            'sq_max_upgraded': round(float(np.max([r['REG_sq'] for r in cur])), 4) if cur else None,
            'sq_no_worse_cov_upgraded': round(float(np.mean([x['REG_sq'] <= 1e-9 for x in cur])), 4) if cur else None,
            'or_max_committed': round(float(np.max([r['REG_or'] for r in cur])), 4) if cur else None,
            'F0_by_seed': [extra[(c, s)]['F0'] for s in SEEDS],
        }
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1906',
           'note': 'tau-free safe-upgrade gate (M6): commit i*(w) iff paired UCB vs status-quo F0 <= 0; else keep F0.',
           'agg': agg, 'per_carrier': per_carrier, 'rows': all_rows}
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print("M2.5 reproduction: EXACT MATCH on all frozen fields (chosen/true_regret/UB_paired/committed).")
    print("AGG:", json.dumps(agg, indent=2))
    for c, v in per_carrier.items():
        print(f"  {c}: {v}")


if __name__ == '__main__':
    main()