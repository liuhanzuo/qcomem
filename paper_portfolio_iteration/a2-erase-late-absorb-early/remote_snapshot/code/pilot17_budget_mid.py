"""pilot17_budget_mid (r782, A4 Gate-B audit M5 fix-v): intermediate-budget arms to create
REAL degrees of freedom for the T3 OU coverage model cov = 1 - exp(-lam * B).

Context: pilot15's four arms sit at B=2.4 (A0/A3) and B=24 (A1/A2). The B=24 pair saturates
(cov~1) for any lam >~ 0.2, so the self-fit lam_hat has effective dof ~ 1 (only the B=2.4
pair pins it). The T1<->T3 seam predicts B_crit ~ 1/lam ~ 3.8. We therefore register two
GD arms (K=1) with window budgets straddling B_crit:
  M1 B~1.2  (eta_win=0.04,  E_win=30*0.04=1.2)   -> predicted cov = 1-exp(-lam*1.2)
  M2 B~6.0  (eta_win=0.20,  E_win=30*0.20=6.0)   -> predicted cov = 1-exp(-lam*6.0)
GD K=1 keeps the budget knob purely in eta (no K, no noise): cleanest test of the scalar-B law.

Preregistered:
  M-ord:  cov(B=1.2) < cov(B=2.4) < cov(B=6.0) < cov(B=24)   (monotone across the seam)
  M-fit:  with lam fixed PARAMETER-FREE at lam_path=0.2023 (pilot11 trapezoid, see
          verify_t3_ou), |pred - measured| < 0.15 on BOTH new arms.
Setup identical to pilot15 (convex ridge logistic, piecewise LR 0.8/0.08 split@120, T=240,
inj@(210,240), 6 seeds, CPU), only window eta differs.
"""
import numpy as np, json, time, sys
sys.path.insert(0, '.')
from pilot7_ablate2x2 import sigmoid
from pilot5_boundary import make_data

t0 = time.time()
lam = 1e-3
T, T_split, INJ = 240, 120, 210
LAM_PATH = 0.202296  # parameter-free, pilot11 trapezoid (verify_t3_ou)

def eta_of(t, eta_win):
    if t < T_split: return 0.8
    if t < INJ: return 0.08
    return eta_win

def train(XA, yA, XD, yD, win, seed, eta_win):
    rng = np.random.default_rng(20_000 + seed)
    w = rng.standard_normal(XA.shape[1]) * 0.05
    for t in range(T):
        eta = eta_of(t, eta_win)
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        z = X @ w; p = sigmoid(z)
        g = X.T @ (p - y) / len(y) + lam * w
        w = w - eta * g
    return w

def acc(w, Xt, yt): return float(np.mean((Xt @ w > 0) == (yt > 0.5)))

arms = {'M1_B1.2': 0.04, 'M2_B6.0': 0.20}
out = {}
for name, eta_win in arms.items():
    fr, cv = [], []
    for seed in range(6):
        XA, yA, XD, yD, Xt, yt = make_data(seed)
        w_nv = train(XA, yA, XD, yD, None, seed, eta_win)
        w_al = train(XA, yA, XD, yD, (0, T), seed, eta_win)
        w_ij = train(XA, yA, XD, yD, (INJ, T), seed, eta_win)
        a_nv, a_al, a_ij = acc(w_nv, Xt, yt), acc(w_al, Xt, yt), acc(w_ij, Xt, yt)
        dmg = a_nv - a_al
        fr.append((a_nv - a_ij) / dmg if dmg > 1e-9 else 0.0)
        sep = np.linalg.norm(w_al - w_nv)
        cv.append(np.linalg.norm(w_ij - w_nv) / sep if sep > 1e-9 else 0.0)
    B = 30 * eta_win  # K=1
    pred = 1 - np.exp(-LAM_PATH * B)
    out[name] = dict(B=B, eta_win=eta_win, frac_med=float(np.median(fr)),
                     cov_med=float(np.median(cv)), pred_cov_param_free=float(pred),
                     resid=float(pred - np.median(cv)),
                     frac_all=[round(x, 3) for x in fr], cov_all=[round(x, 3) for x in cv])

c12, c24, c60, c240 = (out['M1_B1.2']['cov_med'], 0.4695903966563031,
                       out['M2_B6.0']['cov_med'], 0.9987985280140779)
M_ord = bool(c12 < c24 < c60 < c240)
M_fit = bool(max(abs(out['M1_B1.2']['resid']), abs(out['M2_B6.0']['resid'])) < 0.15)
out['verdicts'] = dict(M_ord_monotone_across_seam=M_ord, M_fit_param_free=M_fit,
                       runtime_sec=round(time.time() - t0, 1))
json.dump(out, open('pilot17_budget_mid_out.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
