"""verify_t3_ou (r782 revision; r780 original): numeric check of T3's OU mean-contraction rate.
Predict coverage = 1 - exp(-lam_eff * B), B = K * E_win, for the four pilot15 arms.

r782 fixes (A4 Gate-B audit M5 + m2):
  (i)   cond_i now compares MEASURED cov of equal-budget arms (A0 vs A3, A1 vs A2),
        NOT model-pred vs model-pred (which was vacuously true). Measured deltas:
        A0 0.46959 vs A3 0.46289 (d=0.0067); A1 0.99880 vs A2 0.99856 (d=0.0002).
  (iv)  NEW parameter-free row: lam_path from trapezoidal integration of pilot11's
        shape_sweep (seed0) lambda(delta) over delta in [0,1] -- NO fitting on pilot15.
        Predicts cov(B) = 1-exp(-lam_path*B); compared to measured with prereg 0.15.
        (A4's independent cross-check: lam_path ~= 0.20 -> cov(B=2.4) = 0.384 vs 0.470.)
  (v)   Registered intermediate-budget arms (B~1, B~6, straddling B_crit~3.8) to create
        REAL degrees of freedom -> pilot17_budget_mid.py fills measured cov for these.
  m2:   comment inequality direction fixed (lam_hat = 0.262 > 0.037, not <<).

budgets (window = [210,240), 30 epochs):
  A0 GD base   : K=1,  eta=0.08  -> E_win=2.4,  B=2.4
  A1 GD hightail:K=1,  eta=0.80  -> E_win=24,   B=24
  A2 SGD base  : K=10, eta=0.08  -> E_win=2.4,  B=24
  A3 SGD lowtail:K=10, eta=0.008 -> E_win=0.24, B=2.4
measured cov_med (pilot15): A0 0.4696, A1 0.9988, A2 0.9986, A3 0.4629

Conditions:
  (i)   MEASURED equal-budget arms coincide: |cov_A0-cov_A3|<0.15, |cov_A1-cov_A2|<0.15
  (ii)  coverage monotone in B (on measured + model)
  (iii) single fitted lam_hat fits all four with residual < 0.15 (grid search; REPORT
        effective dof ~= 1 -- the two B=24 arms saturate for any lam >~ 0.2, so only the
        B=2.4 pair pins one scalar; this is a self-fit, kept for continuity)
  (iv)  PARAMETER-FREE lam_path (pilot11 trapezoid) predicts all four with |resid|<0.15
"""
import numpy as np, json

arms = {'A0_GD_base': dict(K=1, eta=0.08, cov=0.4695903966563031),
        'A1_GD_hightail': dict(K=1, eta=0.80, cov=0.9987985280140779),
        'A2_SGD_base': dict(K=10, eta=0.08, cov=0.9985634877020262),
        'A3_SGD_lowtail': dict(K=10, eta=0.008, cov=0.4628924201848913)}
N_EP = 30
for a in arms.values():
    a['E_win'] = N_EP * a['eta']
    a['B'] = a['K'] * a['E_win']

B = np.array([arms[k]['B'] for k in arms])
C = np.array([arms[k]['cov'] for k in arms])

# ---- (iv) parameter-free lam_path from pilot11 shape_sweep_seed0, trapezoid over frac in [0,1]
p11 = json.load(open('pilot11_lambda_traj_out.json'))
rows = p11['shape_sweep_seed0']  # list of {frac, lam, lam_over_lam0}
fracs = np.array([r['frac'] for r in rows]); lams = np.array([r['lam'] for r in rows])
m = (fracs >= 0.0) & (fracs <= 1.0)
lam_path = float(np.trapz(lams[m], fracs[m]))  # = integral_0^1 lambda(delta) ddelta

# ---- (iii) legacy self-fit single lam_hat (kept; flagged dof~1)
lam_grid = np.linspace(1e-4, 2.0, 200000)
best = None
for lam in lam_grid:
    pred = 1 - np.exp(-lam * B)
    rss = float(np.sum((pred - C) ** 2))
    if best is None or rss < best[0]:
        best = (rss, lam)
rss, lam_hat = best
pred = 1 - np.exp(-lam_hat * B)
resid = pred - C

# ---- parameter-free predictions with lam_path
pred_pf = 1 - np.exp(-lam_path * B)
resid_pf = pred_pf - C

# ---- conditions
order = ['A0_GD_base', 'A1_GD_hightail', 'A2_SGD_base', 'A3_SGD_lowtail']
ci = {k: order.index(k) for k in order}
cond_i = (abs(C[ci['A0_GD_base']] - C[ci['A3_SGD_lowtail']]) < 0.15 and
          abs(C[ci['A1_GD_hightail']] - C[ci['A2_SGD_base']]) < 0.15)  # MEASURED equal-B pairs
# cond_ii: monotone in B up to the equal-budget tolerance (equal-B pairs may differ by <0.15,
# already gated by cond_i); require strictly larger-B groups to not drop below smaller-B groups.
uB = np.unique(B)
grp = [np.max(C[B == b]) for b in uB]
cond_ii = bool(np.all(np.diff(grp) >= -1e-9))
cond_iii = bool(np.max(np.abs(resid)) < 0.15)
cond_iv = bool(np.max(np.abs(resid_pf)) < 0.15)

out = dict(
    budgets={k: dict(B=arms[k]['B'], E_win=arms[k]['E_win'], cov_meas=arms[k]['cov'])
             for k in arms},
    lam_hat_self_fit=lam_hat, rss=rss, self_fit_dof=1,
    cov_pred_self_fit={k: float(p) for k, p in zip(arms, pred)},
    lam_path_param_free=lam_path,
    cov_pred_param_free={k: float(p) for k, p in zip(arms, pred_pf)},
    resid_param_free={k: float(r) for k, r in zip(arms, resid_pf)},
    max_abs_resid_param_free=float(np.max(np.abs(resid_pf))),
    cond_i_measured_equalB_equalcov=bool(cond_i),
    cond_ii_monotone=bool(cond_ii),
    cond_iii_single_lam_self_fit=cond_iii,
    cond_iv_param_free_path=cond_iv,
    mid_budget_arms='registered in pilot17_budget_mid.py (B~1, B~6 straddle B_crit~3.8); '
                    'measured cov pending -> fills real dof for cond_iv',
    note=('lam_hat=0.262 (self-fit, dof~1: the two B=24 arms saturate for any lam>~0.2, only '
          'the B=2.4 pair pins the scalar). lam_hat is BETWEEN the near-field '
          'lambda_eff(w*_A)=0.037 and the far-field saturation ~0.51: a path average, '
          'consistent with pilot11 lambda(delta). cond_i now uses MEASURED cov (A0 vs A3 '
          'd=0.0067, A1 vs A2 d=0.0002). cond_iv is the parameter-free cross-check '
          '(lam_path from pilot11 trapezoid, no fitting on pilot15) suggested by A4.'))
json.dump(out, open('verify_t3_ou_out.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
