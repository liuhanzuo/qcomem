"""r1899: controlled operating characteristic of the conditional gate (next-step c).

M3.5 conditional selection gates on FIT-visible CV(hat beta) + whether W is spanning
(large CV AND non-spanning -> convex-minimax; else uniform). Theory (THEORY_MINIMAX
prop uniform-opt): the ONLY driver that breaks uniform optimality is NON-SPANNING W;
CV(hat beta) is a *diagnostic* that tells WHERE the binding high-variance groups are.

This runner makes the gate's operating characteristic explicit with a controlled
synthetic family (G=4, same paired-MPB certificate, budget R, TRUE gap in groups 0,2):

  axis A: deployed set   spanning (uniform + all e_g)  vs  non-spanning (e_0,e_1)
  axis B: CV(hat beta)   low (symmetric beta)  vs  high (groups 0,1 high variance)

Rather than one knife-edge tau (the deterministic counterexample commits at a lucky
draw), we sweep tau and report committed_rate = E[1{UB <= tau}] over (seeds x rows).
This is the robust operating characteristic: the ROC of "committ under a rule".

Prediction:
  - non-spanning: minimax committed_rate >= uniform at every tau (zeroing study on the
    binding group never hurts the bound when only that group's weight is active);
    high CV widens the gap (the binding group is exactly the high-variance one).
  - spanning: uniform is (near-)optimal; minimax effectively == uniform (degrades to
    near-equal spread since every group binds under max over W).
  => gate "if non-spanning -> minimax else uniform" is safe (never worse than the
     equal-spread baseline) and effective (strict gain on non-spanning).

We also stamp the real-carrier placement (r1896/r1897): real spanning grid keeps
uniform=maximin; real non-spanning high-var probes use minimax -> matches this OC.

PROJECT A2_SAFE_MODEL_RANKING_SUBGROUP_MIX ROUND r1899 Pure CPU front.
"""
import json, os, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'results', 'SUBGMIX_GATE_OC_R1899.json')
DELTA = 0.10
G = 4
R = 1200
M = 2
N_CAL_PER_GROUP = 5000
SEEDS = list(range(20))
TAUS = [0.02, 0.03, 0.04, 0.06, 0.08, 0.12]
TRUE_DELTA = np.array([0.02, 0.0, 0.02, 0.0])


def dcell(M, G):
    return DELTA / (M * (M - 1) * max(G, 1))


def make_pool(beta_g, delta_g, seed):
    a = max(0.0, (beta_g ** 2 + delta_g) / 2.0)
    b = max(0.0, (beta_g ** 2 - delta_g) / 2.0)
    rng = np.random.RandomState(seed)
    u = rng.rand(N_CAL_PER_GROUP)
    d = np.zeros(N_CAL_PER_GROUP)
    d[u < a] = 1.0
    d[(u >= a) & (u < a + b)] = -1.0
    return d


def mpb_ucb(d_sub, dc):
    n = d_sub.size
    if n == 0:
        return np.inf
    mu = d_sub.mean()
    s = d_sub.std(ddof=1) if n > 1 else 0.0
    mX = (mu + 1.0) / 2.0
    vX = (s ** 2) / 4.0
    L = np.log(2.0 / dc)
    ubX = mX + np.sqrt(2.0 * vX * L / n) + 7.0 * L / (3.0 * (n - 1)) if n > 1 \
        else mX + np.sqrt(L / 2.0)
    return 2.0 * ubX - 1.0


def waterfill_abs(eff, R):
    big = np.maximum(eff ** (2.0 / 3.0), 1e-12)
    ng = np.maximum(np.floor(big / big.sum() * R), 1).astype(int)
    while ng.sum() < R:
        ng[np.argmax(big)] += 1
    while ng.sum() > R:
        ng[np.argmax(big)] -= 1
    assert ng.sum() == R and (ng >= 1).all() and (ng <= N_CAL_PER_GROUP).all()
    return ng


def cv_beta(b):
    nz = b > 1e-6
    return float(b[nz].std() / b[nz].mean()) if nz.mean() > 0 and abs(b[nz].mean()) > 1e-12 else 0.0


def run_cell(cv_level, spanning, seed):
    beta = np.array([0.06, 0.06, 0.06, 0.06]) if cv_level == 'low' \
        else np.array([0.12, 0.12, 0.02, 0.02])
    if spanning:
        W = np.vstack([np.ones(G) / G, np.eye(G)])
    else:
        W = np.eye(G)[[0, 1]]
    eff = np.max(W * beta, axis=0)
    ng_uniform = np.array([R // G] * G)
    ng_uniform[:R - ng_uniform.sum()] += 1
    ng_mm = waterfill_abs(eff, R)
    dc = dcell(M, G)
    pools = {g: make_pool(beta[g], TRUE_DELTA[g], seed * 100 + g) for g in range(G)}
    row_ub = {'uniform': [], 'minimax': []}
    row_safe = {'uniform': [[], []], 'minimax': [[], []]}   # per-rule list per row of bool
    for rule, ng in (('uniform', ng_uniform), ('minimax', ng_mm)):
        ucb = np.array([mpb_ucb(pools[g][:ng[g]], dc) for g in range(G)])
        for r in range(W.shape[0]):
            w = W[r]
            ub = float(np.dot(w, ucb))
            true_reg = float(np.dot(w, TRUE_DELTA))
            row_ub[rule].append(ub)
            row_safe[rule][0].append(r)
            row_safe[rule][1].append(bool(true_reg <= max(TAUS) + 1e-9))
    return row_ub, row_safe


def main():
    t0 = time.time()
    rows = []
    for spanning in (True, False):
        for cv_level in ('low', 'high'):
            unif = []
            mm = []
            cv = None
            for seed in SEEDS:
                u, s = run_cell(cv_level, spanning, seed)
                unif += u['uniform']
                mm += u['minimax']
                cv = s  # cv same across seeds (deterministic betas)
            unif = np.array(unif)
            mm = np.array(mm)
            # TRUE_DELTA max=0.02, so every tau>=0.02 has true_reg<=tau by construction; a
            # committed row is sound for every tau in the sweep.
            n_w_rows = 5 if spanning else 2
            curve_u = [float((unif <= t).mean()) for t in TAUS]
            curve_m = [float((mm <= t).mean()) for t in TAUS]
            # signed gap area over the tau grid: positive means minimax > uniform.
            gap = float(np.mean(np.array(curve_m) - np.array(curve_u)))
            worst_axis = None
            worst_mmu = 0.0
            for (t, cuv, cmv) in zip(TAUS, curve_u, curve_m):
                if cmv - cuv < worst_mmu:
                    worst_mmu = cmv - cuv
                    worst_axis = t
            rows.append({'deployed': 'spanning' if spanning else 'non-spanning',
                         'cv_level': cv_level, 'cv_beta': cv_beta(
                             np.array([0.06] * 4 if cv_level == 'low' else [0.12, 0.12, 0.02, 0.02])),
                         'n_rows_seeds': len(unif), 'n_w_rows': n_w_rows,
                         'curve_uniform': curve_u, 'curve_minimax': curve_m,
                         'taus': TAUS, 'signed_area_minimax_over_uniform': round(gap, 4),
                         'worst_minimax_deficit_at_tau': worst_axis,
                         'worst_minimax_deficit': round(worst_mmu, 4)})
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1899',
           'kind': 'conditional_gate_operating_characteristic',
           'delta': DELTA, 'G': G, 'R': R, 'tau_sweep': TAUS, 'n_seeds': len(SEEDS),
           'prediction': ('non-spanning => minimax committed_rate >= uniform at every tau; '
                          'high CV widens the gap; spanning => minimax == uniform. '
                          'Gate: non-spanning->minimax else uniform is safe+effective.'),
           'rows': rows, 'runtime_s': round(time.time() - t0, 2)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('tau_sweep', TAUS)
    print(f"{'deployed':13}{'cv':>5} {'sarea'}  curves(uniform>>minimax)")
    for r in rows:
        cu = ' '.join(f"{x:.2f}" for x in r['curve_uniform'])
        cm = ' '.join(f"{x:.2f}" for x in r['curve_minimax'])
        print(f"{r['deployed']:13}{r['cv_level']:>5} {r['signed_area_minimax_over_uniform']:+.3f}  "
              f"U[{cu}]  M[{cm}]")
    print('runtime_s', out['runtime_s'])


if __name__ == '__main__':
    main()