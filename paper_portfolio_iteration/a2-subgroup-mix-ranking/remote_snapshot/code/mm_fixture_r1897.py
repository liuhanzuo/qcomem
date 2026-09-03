"""r1897 numerical fixture validating the corrected convex-minimax.

Verifies, across several (M,G) shapes and beta/w combinations:
  (1) fixed-mu water-fill n_g = R a_g(mu)^{2/3}/S, value = S^{3/2}/sqrt(R);
  (2) value = 2 * lambda * R (MGR-reconciled identity), homogeneous R^{-1/2};
  (3) STRONG DUALITY: P* = d* = R^{-1/2}[max_u S(u)]^{3/2} over the ENGINE simplex
      (engines = (model j, deployed mixture w) pairs), within tol on the continuous primal;
  (4) complementary slackness: active set = {j: u*_j>0} (engines attaining the max at n*),
      NOT all candidates; verified on cases where a strict subset binds.
The primal is exact on the integer allocation n; the continuous-relaxation primal isolates
the discreteness cost.

PROJECT A2_SAFE_MODEL_RANKING_SUBGROUP_MIX ROUND r1897  Pure CPU front.
"""
import json, os, numpy as np
from minimax_core_r1897 import minimax_solve

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', 'MM_FIXTURE_R1897.json')

# For each case, build the ENGINE matrix C of shape (K, G) with engines (model j, mixture w).
CASES = [
    {'name': 'A2_asym_one_binding',
     'C': np.array([[0.100, 0.100, 0.020, 0.020],   # (j0, w=e_0): only g0 binds
                    [0.000, 0.000, 0.000, 0.000]]),  # (j1, e_0): dominated
     'w': None, 'R': 1200},
    {'name': 'B_symmetric',
     'C': np.array([[0.05, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.05, 0.05]]) * 1.0,
     'w': None, 'R': 1200},
    {'name': 'C_binding_pair',
     'C': np.array([[0.100, 0.010, 0.010, 0.010],   # engine0 binds g0
                    [0.010, 0.100, 0.010, 0.010],   # engine1 binds g1 (joins active set)
                    [0.005, 0.005, 0.080, 0.005]]), # engine2 binds g2
     'w': None, 'R': 1200},
    {'name': 'D_three_models',
     'C': np.array([[0.100, 0.100, 0.020],
                    [0.050, 0.120, 0.040],
                    [0.110, 0.090, 0.010],
                    [0.080, 0.080, 0.080]]),
     'w': None, 'R': 800},
]


def build_C(beta_list, w_list):
    Crows = []
    for w in w_list:
        for j in range(len(beta_list)):
            Crows.append(np.array(w, float) * np.array(beta_list[j], float))
    return np.array(Crows)


def exact_primal(n, C):
    n = np.asarray(n, float)
    return float(max(sum(C[k, g] / np.sqrt(n[g]) for g in range(C.shape[1]))
                     for k in range(C.shape[0])))


def cont_primal(C, u, R):
    a = np.maximum(u @ C, 1e-12)
    S = float(np.sum(a ** (2.0 / 3.0)))
    n = R * a ** (2.0 / 3.0) / S
    return exact_primal(n, C), S


def main():
    res, checks = {}, {}
    for cs in CASES:
        C = cs['C'] if cs['C'] is not None else build_C(cs['beta'], cs['w'])
        sol = minimax_solve(C, cs['R'], n_min=1, seed=7)
        Pr = exact_primal(sol['n_g'], C)
        dstar = float(sol['S_star'] ** 1.5 / np.sqrt(cs['R']))
        Pr_cont, S_cont = cont_primal(C, np.asarray(sol['u_star']), cs['R'])
        gap_cont = abs(Pr_cont - dstar) / max(1e-12, abs(dstar))
        gap_int = abs(Pr - dstar) / max(1e-12, abs(dstar))
        ident = abs(sol['value_cont'] - sol['two_lambda_R']) / max(1e-12, abs(sol['value_cont']))
        n = np.asarray(sol['n_g'], float)
        Phi = np.array([sum(C[k, g] / np.sqrt(n[g]) for g in range(C.shape[1]))
                        for k in range(C.shape[0])])
        active = sol['active_set']
        act_phi = [Phi[k] for k in active]
        on_act = max(act_phi) - min(act_phi) if active else 0.0
        offc = [Phi[k] for k in range(C.shape[0]) if k not in active]
        slack = (max(offc) - max(act_phi)) if offc and active else 0.0
        res[cs['name']] = {'sol': sol, 'primal': float(Pr), 'primal_cont': float(Pr_cont),
                           'dual_dstar': float(dstar),
                           'gap_cont_rel': float(gap_cont), 'gap_int_rel': float(gap_int),
                           'identity_err': float(ident),
                           'Phi': Phi.tolist(), 'active_set': active,
                           'active_Phi_spread': float(on_act),
                           'inactive_slack': float(slack),
                           'K': C.shape[0], 'G': C.shape[1]}
        checks[cs['name']] = {
            'cont_strong_duality_gap<=1e-6': bool(gap_cont <= 1e-6),
            'int_gap_rel': float(gap_int),
            'value_eq_2lambdaR<=1e-9': bool(ident <= 1e-9),
            'active_set==worst_engines': bool(
                (not active) or (abs(on_act) <= 1e-6 and slack <= 1e-6)),
        }

    # homogeneity across R for case C_binding_pair
    homog = {}
    Rvals = np.array([400, 800, 1200, 2400, 4800])
    C = CASES[2]['C']
    for Rv in Rvals:
        s = minimax_solve(C, int(Rv), seed=7)
        homog[str(Rv)] = {'S_star': s['S_star'], 'value_cont': s['value_cont'],
                          'v_sqrtR': s['value_cont'] * np.sqrt(Rv), 'lambda': s['lambda']}
    homog_spread = max(x['v_sqrtR'] for x in homog.values()) - \
        min(x['v_sqrtR'] for x in homog.values())

    out = {'round': 'r1897', 'kind': 'fixture_strong_duality_and_homogeneity',
           'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX',
           'cases': res, 'checks': checks, 'homogeneity_ACROSS_R': homog,
           'homog_v_sqrtR_spread': float(homog_spread),
           'identity': 'value = 2*lambda*R; value ~ R^{-1/2}; P*=d*=R^{-1/2}[max_u S(u)]^{3/2}'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")
    for name, c in checks.items():
        print(f"{name}: cont_gap<1e-6={c['cont_strong_duality_gap<=1e-6']} "
              f"int_gap={c['int_gap_rel']:.2e} v=2lR<1e-9={c['value_eq_2lambdaR<=1e-9']} "
              f"active_correct={c['active_set==worst_engines']}")
        r = res[name]
        print(f"   active_set={r['active_set']} spread={r['active_Phi_spread']:.2e} "
              f"inactive_slack={r['inactive_slack']:.2e} n_g={r['sol']['n_g']}")
    print(f"homogeneity: v_sqrtR spread = {homog_spread:.3e} over R={Rvals.tolist()}")


if __name__ == '__main__':
    main()