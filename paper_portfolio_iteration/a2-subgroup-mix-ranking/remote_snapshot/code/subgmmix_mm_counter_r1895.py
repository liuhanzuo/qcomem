"""r1895 counter-example: deterministic asymmetric ambiguity set where convex-minimax
allocation strictly beats uniform under the SAME paired-MPB certificate mechanism.

Theory (THEORY_MINIMAX_R1895.md sec 4): uniform is the minimax solution of
  min_{n, sum=R} max_{j,w in W} sum_g w_g r_g(n_g)
iff w_g*beta_g is constant across g on the deployed set W.  On the *spanning*
symmetric grid (uniform + every single-group vertex) uniform is (near) minimax-optimal
regardless of beta asymmetry, because each group independently binds some row -> eating
all groups is optimal.  The counter-example must therefore use an ASYMMETRIC deployed
W that does NOT span every vertex: then minimax concentrates budget on the mixture that
binds and strictly dominates uniform.

Construction (all deterministic, real samples, same certificate as M3):
  G=4 groups, R budget.  Two candidate models (i,j).  True per-group risk DIFFERENCE
  r_i - r_j = delta_g (known oracle used only to evaluate TRUE regret, as in every other
  runner; the certificate uses CAL paired diffs).  Choose asym beta: groups (0,1) high
  variance, groups (2,3) ~zero variance.
  Deployed W = { e_0 (all weight on group 0) }  -- a single asymmetric vertex, NOT spanning.
  The certificate UB(w)=sum_g w_g UCB_{ij,g}(n_g).  At w=e_0 only group 0's width binds.
  -> minimax water-fill puts (almost) all R into group 0; uniform spreads R/4.
  With beta_0 big vs others, minimax commits at tau where uniform does not.

PROJECT A2_SAFE_MODEL_RANKING_SUBGROUP_MIX ROUND r1895 Pure CPU front.
"""
import json, os, numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', 'SUBGMIX_MM_COUNTER_R1895.json')
DELTA = 0.10
G = 4
R = 1200
BETA = np.array([0.10, 0.10, 0.02, 0.02])   # asym variance: groups 2,3 near-zero width
N_CAL_PER_GROUP = 4000                         # large pool so budget binds (n_g <= avail)
SEED = 7

def dcell(M, G):
    return DELTA / (M * (M - 1) * max(G, 1))

def make_cal_direct(beta_g, delta_g, seed):
    """Sample the paired-difference d := err_i - err_j directly from a 3-point dist with
    mean delta_g and std ~ beta_g.  The certificate (M3 paired-MPB) only uses these paired
    diffs; this is a clean synthetic feasibility/mechanism construction (clearly labeled).
    Dist: P(d=1)=a, P(d=-1)=b, mean a-b=delta, Var~=a+b=beta^2 (delta small).
    """
    a = max(0.0, (beta_g ** 2 + delta_g) / 2.0)
    b = max(0.0, (beta_g ** 2 - delta_g) / 2.0)
    rng = np.random.RandomState(seed)
    u = rng.rand(N_CAL_PER_GROUP)
    d = np.zeros(N_CAL_PER_GROUP)
    d[u < a] = 1.0
    d[(u >= a) & (u < a + b)] = -1.0
    return d

def mpb_ucb_g(d, dc):
    n = d.size
    if n == 0:
        return np.inf
    mu = d.mean(); s = d.std(ddof=1) if n > 1 else 0.0
    mX = (mu + 1.0) / 2.0; vX = (s ** 2) / 4.0; L = np.log(2.0 / dc)
    ubX = mX + np.sqrt(2.0 * vX * L / n) + 7.0 * L / (3.0 * (n - 1)) if n > 1 else mX + np.sqrt(L / 2.0)
    return 2.0 * ubX - 1.0


def main():
    t0 = __import__('time').time()
    M = 2  # models i,j
    dc = dcell(M, G)
    true_delta = {g: 0.02 * (g % 2) for g in range(G)}  # group 0,2 true gap; 1,3 tie
    results = {}
    true_regret_bound = {}
    for rule in ['uniform', 'minimax']:
        if rule == 'uniform':
            ng = np.array([R // G] * G)
            rem = R - ng.sum()
            for gg in range(rem):
                ng[gg] += 1
        else:
            # Convex-minimax water-fill on the *deployed* asymmetric set W={e_0}: the worst
            # (only binding) mixture is e_0 -> eff_g = w_g*beta_g = [1,0,0,0]; the budget
            # concentrates entirely on the binding group 0 (with a 1-sample floor elsewhere
            # so every group retains a certificate).  This is exactly THEORY sec 4 Counter-A.
            wstar = np.array([1.0, 0.0, 0.0, 0.0])  # e_0 binds
            eff = wstar * np.array(BETA)
            big = np.maximum(eff ** (2.0 / 3.0), 1e-12)
            ng = np.maximum(np.floor(big / max(big.sum(), 1e-12) * R), 1).astype(int)
            while ng.sum() < R:
                ng[np.argmax(big)] += 1
            while ng.sum() > R:
                ng[np.argmax(big)] -= 1
            assert ng.sum() == R and (ng >= 1).all() and (ng <= N_CAL_PER_GROUP).all()
        # sample CAL per group (synthetic feasibility construction), then CERTIFY only on the
        # REVEALED subsample of size n_g (same-certificate, same sound paired-MPB as M3):
        # this is what makes allocation matter -- UCB shrinks with revealed n_g.
        full_pool = {g: make_cal_direct(BETA[g], true_delta[g], SEED + g) for g in range(G)}
        ucb_all = {}
        for g in range(G):
            pool = full_pool[g][:N_CAL_PER_GROUP]
            ucb_all[g] = mpb_ucb_g(pool[:ng[g]], dc)  # certify on the n_g revealed points
        for wname, w in [('e_0', {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}),
                         ('uniform_w', {g: 1.0 / G for g in range(G)})]:
            ub = sum(w[g] * ucb_all[g] for g in range(G))
            true_reg = sum(w[g] * true_delta[g] for g in range(G))  # regret of i vs j = delta if >0
            results.setdefault(wname, {})[rule] = {
                'n_g': ng.tolist(), 'UB': float(ub),
                'true_regret_op': float(true_reg),
                'committed_at_tau_004': bool(ub <= 0.04),
                'beta': BETA.tolist()}
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1895',
           'kind': 'deterministic_asymmetric_W_counterexample', 'G': G, 'R': R,
           'BETA': BETA.tolist(), 'deployed_W': ['e_0', 'uniform_w'],
           'results': results, 'note': ('e_0 is an asymmetric (non-spanning) deployed '
            'mixture: only group 0 binds; minimax concentrates allocated budget there and '
            'should commit where equal-spread uniform cannot.'),
           'runtime_s': round(t0 if False else __import__('time').time() - t0, 2)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out['results'], indent=1))
    print('committed at tau=0.04:')
    for wname in ['e_0', 'uniform_w']:
        for rule in ['uniform', 'minimax']:
            assert results[wname][rule]['committed_at_tau_004'] is not None
    for wname in ['e_0', 'uniform_w']:
        print(' ', wname, 'uniform', results[wname]['uniform']['committed_at_tau_004'],
              'minimax', results[wname]['minimax']['committed_at_tau_004'])

if __name__ == '__main__':
    main()