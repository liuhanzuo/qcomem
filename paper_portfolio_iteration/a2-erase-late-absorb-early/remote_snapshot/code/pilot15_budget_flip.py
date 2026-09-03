"""pilot15 (r780): decisive falsifiable test of T3-REVISED (contraction-budget traversal)
vs T3-OLD (optimizer identity / noise floor).

Revised theory: absorption within an inject window = traversal toward the window
attractor, controlled by total contraction budget  x ~ c_path * K * sum(eta_win),
where K = updates per epoch. Optimizer identity (GD vs SGD) and gradient noise are
NOT the causal variable: K=1 (any GD) -> primacy anchor; K>1 or large eta_win -> recency.

diag8 (r780) killed the noise-floor mechanism: stationary endpoint spread around the
clean attractor is ~0.05-0.08, i.e. 30-50x SMALLER than the attractor separation 2.61,
and the OU scaling Tr(Cov) ~ eta^2 K sigma^2/mu fails by 45x across bs.

Four arms (convex ridge logistic, piecewise LR 0.8/0.08 split@120, T=240, inj@(210,240),
6 seeds, CPU):
  A0 GD base      : eta_lo=0.08 in window; budget x0 (anchor, expect primacy)
  A1 GD hightail  : eta=0.80 in window (10x budget, K=1); expect FLIP TO RECENCY
  A2 SGD base bs128: eta_lo=0.08, K=10; expect recency (anchor)
  A3 SGD lowtail  : eta_lo=0.008 in window (budget = x0*0.1*10 ~ x0), noise structure
                    UNCHANGED (fresh perm, same bs); expect FLIP TO PRIMACY.

Preregistered (median over 6 seeds):
  P1: frac(A1) >= 3 * frac(A0)          (GD flips to recency when budget raised)
  P2: frac(A3) <= 0.5 * frac(A2)        (SGD flips to primacy when budget cut, noise on)
  P3: frac(A3)/frac(A0) in [0.3, 3]     (equal budget => equal absorption, any optimizer)
  Secondary (geometry, margin-pollution-free):
  cov = ||w_inj - w_never|| / ||w_always - w_never||  reported for all arms.
"""
import numpy as np, json, time, sys
sys.path.insert(0, '.')
from pilot7_ablate2x2 import sigmoid
from pilot5_boundary import make_data

t0 = time.time()
lam = 1e-3
T, T_split, INJ = 240, 120, 210
BS = 128

def eta_of(t, arm):
    if t < T_split: return 0.8
    if t < INJ: return 0.08
    # window [INJ, T)
    if arm == 'hightail': return 0.8
    if arm == 'lowtail': return 0.008
    return 0.08

def train(XA, yA, XD, yD, win, seed, sgd, arm):
    rng = np.random.default_rng(20_000 + seed)
    w = rng.standard_normal(XA.shape[1]) * 0.05
    for t in range(T):
        eta = eta_of(t, arm)
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        if sgd:
            perm = rng.permutation(len(y))
            for i in range(0, len(y), BS):
                idx = perm[i:i + BS]
                z = X[idx] @ w; p = sigmoid(z)
                g = X[idx].T @ (p - y[idx]) / len(idx) + lam * w
                w = w - eta * g
        else:
            z = X @ w; p = sigmoid(z)
            g = X.T @ (p - y) / len(y) + lam * w
            w = w - eta * g
    return w

def acc(w, Xt, yt): return float(np.mean((Xt @ w > 0) == (yt > 0.5)))

arms = {'A0_GD_base': (False, 'base'), 'A1_GD_hightail': (False, 'hightail'),
        'A2_SGD_base': (True, 'base'), 'A3_SGD_lowtail': (True, 'lowtail')}
out = {}
for name, (sgd, arm) in arms.items():
    fr, cv = [], []
    for seed in range(6):
        XA, yA, XD, yD, Xt, yt = make_data(seed)
        w_nv = train(XA, yA, XD, yD, None, seed, sgd, arm)
        w_al = train(XA, yA, XD, yD, (0, T), seed, sgd, arm)
        w_ij = train(XA, yA, XD, yD, (INJ, T), seed, sgd, arm)
        a_nv, a_al, a_ij = acc(w_nv, Xt, yt), acc(w_al, Xt, yt), acc(w_ij, Xt, yt)
        dmg = a_nv - a_al
        fr.append((a_nv - a_ij) / dmg if dmg > 1e-9 else 0.0)
        sep = np.linalg.norm(w_al - w_nv)
        cv.append(np.linalg.norm(w_ij - w_nv) / sep if sep > 1e-9 else 0.0)
    out[name] = dict(frac_med=float(np.median(fr)), cov_med=float(np.median(cv)),
                     frac_all=[round(x, 3) for x in fr], cov_all=[round(x, 3) for x in cv])

f = {k: out[k]['frac_med'] for k in out}
P1 = f['A1_GD_hightail'] >= 3 * f['A0_GD_base']
P2 = f['A3_SGD_lowtail'] <= 0.5 * f['A2_SGD_base']
r = f['A3_SGD_lowtail'] / max(f['A0_GD_base'], 1e-9)
P3 = 0.3 <= r <= 3.0
out['verdicts'] = dict(P1_GD_flip_recency=bool(P1), P2_SGD_flip_primacy=bool(P2),
                       P3_equal_budget_equal_absorption=bool(P3), ratio_A3_over_A0=float(r),
                       runtime_sec=round(time.time() - t0, 1))
json.dump(out, open('pilot15_budget_flip_out.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
