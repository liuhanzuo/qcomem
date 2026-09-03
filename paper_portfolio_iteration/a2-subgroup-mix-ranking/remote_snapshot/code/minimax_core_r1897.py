"""r1897 corrected convex-minimax core.

MGR instruction 593c907d2ccd fixes r1895's dual: the old d* = lambda*R was WRONG-dimensioned
(lambda already carries R^{-3/2}).  Correct fixed-mu solution and strong dual:

  a_g(mu) = sum_j mu_j w_jg beta_jg,   S(mu) = sum_g a_g(mu)^{2/3}
  n_g(mu) = R * a_g(mu)^{2/3} / S(mu)       (fixed mu, budget sum_g n_g = R)
  V(mu)   = S(mu)^{3/2} / sqrt(R)           (value;  homogeneous R^{-1/2})
  lambda  = (1/2) S(mu)^{3/2} R^{-3/2}  =>  V(mu) = 2 lambda R

Mini-max over the deployed mixtures AND the worst candidate mixing (joint convex-vs-concave
h(n,mu) = sum_{j,g} mu_j w_jg beta_jg n_g^{-1/2}); Sion strong duality:

  P* = d* = R^{-1/2} * [ max_{mu in Delta_M} S(mu) ]^{3/2}

S(mu) is CONCAVE on Delta_M (each a_g(mu)^{2/3} is concave-increasing of a linear form, x^p,
0<p<1).  So max_mu S(mu) is a convex program; solved by projected (simplex) gradient ascent.
ACTIVE SET by complementary slackness: only {j: mu*_j>0} (candidates attaining the max at n*),
never "all candidates active".

PROJECT A2_SAFE_MODEL_RANKING_SUBGROUP_MIX ROUND r1897  Pure CPU.
"""
import numpy as np


def simplex_project(v):
    """Euclidean projection of v onto the probability simplex (Duchi et al. 2008)."""
    srt = np.sort(v)[::-1]
    css = np.cumsum(srt) - 1.0
    rho = np.nonzero(srt * np.arange(1, srt.size + 1) > css)[0][-1]
    th = css[rho] / (rho + 1)
    return np.maximum(v - th, 0.0)


def minimax_solve(C, R, n_min=1, max_iter=4000, seed=0, u0_fix=None):
    """Solve the corrected convex-minimax over (n, mu) jointly.

    C : (K, G) engine coefficient matrix; engine k=(model j, mixture w) has
        per-group width coefficient C[k,g] = w_g * beta_{j,g}.  The ambiguity is over
        ALL engines (worst model j AND worst deployed mixture w), so mu mixes over K.
    u0_fix : None (solve max_concave S) or a fixed mixing (e.g. uniform, heuristic
        "all engines active" ablation / old all-candidates-active).
    Returns dict with u*, S*, n_g, value, lambda, active set, dual/primal check.
    """
    C = np.asarray(C, float)
    K, G = C.shape
    Rf = float(R)

    def a(u):
        return np.maximum(u @ C, 1e-12)        # (G,)

    def S(u):
        return float(np.sum(a(u) ** (2.0 / 3.0)))

    def gS(u):
        aa = a(u)
        return ((2.0 / 3.0) * aa ** (-1.0 / 3.0)) @ C.T   # dS/du, (K,)

    if u0_fix is not None:
        u = np.asarray(u0_fix, float); u = u / u.sum()
        Sbest = S(u)
        _ = gS(u)
    else:
        rng = np.random.RandomState(seed)
        Sbest, ubest = -np.inf, None
        cands = [np.full(K, 1.0 / K)] + [rng.dirichlet(np.ones(K)) for _ in range(5)]
        for u0 in cands:
            u = np.array(u0, float)
            mom = np.zeros(K)
            for it in range(max_iter):
                g = gS(u)
                scale = max(1e-3, float(np.abs(g).max()))
                mom = 0.75 * mom + 0.25 * (g / scale)
                u = simplex_project(u + 0.6 * mom / (1.0 + it * 0.005))
                Su = S(u)
                if Su > Sbest:
                    Sbest, ubest = Su, u.copy()
        u = ubest
    aa = a(u)
    raw = Rf * aa ** (2.0 / 3.0) / Sbest
    n = np.maximum(np.floor(raw), n_min).astype(int)
    order = np.argsort(-aa)
    rem = int(R) - int(n.sum()); i = 0
    while rem > 0 and i < 100000:
        g = int(order[i % G])
        n[g] += 1; rem -= 1; i += 1
    while int(n.sum()) > int(R):
        for gg in order:
            if n[gg] > n_min:
                n[gg] -= 1; break
        i += 1
    n = np.maximum(n, n_min)
    value = float(np.sum(aa * n.astype(float) ** (-0.5)))
    value_cont = float(Sbest ** 1.5 / np.sqrt(Rf))
    lam = 0.5 * Sbest ** 1.5 * Rf ** (-1.5)
    return {'u_star': u.tolist(), 'S_star': float(Sbest), 'n_g': n.tolist(),
            'value': value, 'value_cont': value_cont,
            'lambda': float(lam), 'two_lambda_R': float(2.0 * lam * Rf),
            'active_set': [int(j) for j in range(K) if u[j] > 1e-6],
            'K': K, 'G': G}