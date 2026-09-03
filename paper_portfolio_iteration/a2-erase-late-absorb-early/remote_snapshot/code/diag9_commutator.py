"""diag9_commutator (r780): tighten T1 proof Step 2 (the H_t≈const approximation).

T1's proof skeleton linearizes GD as  w_{t+1}-w* = (I - eta_t H_t)(w_t - w*)  and then
approximates the operator product  M = Pi_t (I - eta_t H_t)  by its CONSTANT-H form
Pi_t (I - eta_t H), whose first-order term depends only on E = sum eta_t. The gap is the
TIME-VARIATION of H_t = H(w_t) along the trajectory (the "commutator" error). This is the
weakest link an auditor would attack.

We quantify three damage/terminal-state cross-schedule spreads at matched E, in the
ALWAYS-ON convex+GD cell (the T1 object), at a starved budget where invariance is NOT
trivially "converged-to-attractor":

  (i) EXACT nonlinear trajectory:        damage_range_exact   = spread across shapes
  (ii) FROZEN-H (constant H = H(w*_AD)): damage_range_frozen  = spread across shapes
        -> isolates the pure schedule-shape / operator-ordering effect with H constant
  (iii) H-variation contribution estimate = damage_range_exact - damage_range_frozen

Also report the operator-level quantity the proof bounds:
  Sigma2 = sum_{s<t} eta_s eta_t   for each shape (the O(eta^2) cross-term coefficient),
and check the monotone relation  damage_range  ~  |Sigma2(shape) - Sigma2(reference)|.

Preregistered:
  C1: damage_range_frozen <= damage_range_exact  (freezing H can only reduce spread;
      the residual exact-minus-frozen is the H-variation / linearization error, which the
      proof must show is small).
  C2: damage_range_exact <= 0.01 (T1 invariance holds at this budget).
  C3 (report): rank correlation between |Delta Sigma2| and per-shape damage deviation.
"""
import numpy as np, json, time, sys
sys.path.insert(0, '.')
from pilot7_ablate2x2 import sigmoid
from pilot5_boundary import make_data

t0 = time.time()
lam = 1e-3
T, T_split = 240, 120


def hessian(w, XA, yA, XD, yD):
    """Gauss-Newton + ridge Hessian of the always-on (A∪D) objective at w."""
    X = np.concatenate([XA, XD])
    z = X @ w
    q = sigmoid(z) * (1.0 - sigmoid(z))
    H = (X * q[:, None]).T @ X / len(X)
    H += lam * np.eye(X.shape[1])
    return H


def train_exact(XA, yA, XD, yD, ETA, w0):
    w = w0.copy()
    for t in range(len(ETA)):
        X = np.concatenate([XA, XD]); y = np.concatenate([yA, yD])
        z = X @ w; p = sigmoid(z)
        g = X.T @ (p - y) / len(y) + lam * w
        w = w - ETA[t] * g
    return w


def train_frozen(XA, yA, XD, yD, ETA, w0, w_star, g_star):
    """Linearized GD with H frozen at H(w*):  w <- w - eta*(H@(w-w*) + g*)."""
    H = hessian(w_star, XA, yA, XD, yD)
    w = w0.copy()
    for t in range(len(ETA)):
        w = w - ETA[t] * (H @ (w - w_star) + g_star)
    return w


def acc(w, Xt, yt): return float(np.mean((Xt @ w > 0) == (yt > 0.5)))


def sched3(shape, E, T=240, split=120):
    if shape == 'flat':
        return np.full(T, E / T)
    eh, el = (0.8, 0.08) if shape == 'base' else (0.6, 0.28)
    sc = E / (eh * split + el * (T - split))
    return np.array([eh * sc if t < split else el * sc for t in range(T)])


def sigma2(ETA):
    return float(sum(ETA[s] * ETA[t] for s in range(len(ETA)) for t in range(s + 1, len(ETA))))


# starved budget: 5% of full E (invariance NOT trivial; verify_t1_2 found range~1e-4 here)
E_full = 0.8 * 120 + 0.08 * 120
E = E_full * 0.05
shapes = ['base', 'flat', 'mild']
ETAs = {s: sched3(s, E) for s in shapes}
S2 = {s: sigma2(ETAs[s]) for s in shapes}

res_exact = {s: [] for s in shapes}
res_frozen = {s: [] for s in shapes}
for seed in range(6):
    XA, yA, XD, yD, Xt, yt = make_data(seed)
    w0 = np.random.default_rng(1000 + seed).standard_normal(40) * 0.05
    # reference points
    ETA_big = np.full(400, 0.3)
    X = np.concatenate([XA, XD]); y = np.concatenate([yA, yD])
    w_star = train_exact(XA, yA, XD, yD, ETA_big, w0)          # w*_AD
    z = X @ w_star; g_star = X.T @ (sigmoid(z) - y) / len(y) + lam * w_star  # ~0
    # never arm for damage reference
    ETA_nv = np.full(400, 0.3)
    Xn, yn = XA, yA
    w_nv = w0.copy()
    for t in range(len(ETA_nv)):
        zn = Xn @ w_nv; pn = sigmoid(zn)
        w_nv = w_nv - ETA_nv[t] * (Xn.T @ (pn - yn) / len(yn) + lam * w_nv)
    a_nv = acc(w_nv, Xt, yt)
    for s in shapes:
        w_ex = train_exact(XA, yA, XD, yD, ETAs[s], w0)
        w_fr = train_frozen(XA, yA, XD, yD, ETAs[s], w0, w_star, g_star)
        res_exact[s].append(a_nv - acc(w_ex, Xt, yt))
        res_frozen[s].append(a_nv - acc(w_fr, Xt, yt))

med = lambda v: float(np.median(v))
dmg_exact = {s: med(res_exact[s]) for s in shapes}
dmg_frozen = {s: med(res_frozen[s]) for s in shapes}
rng_exact = max(dmg_exact.values()) - min(dmg_exact.values())
rng_frozen = max(dmg_frozen.values()) - min(dmg_frozen.values())

# per-shape deviation from the flat reference, vs |Delta Sigma2|
dev_exact = {s: abs(dmg_exact[s] - dmg_exact['flat']) for s in shapes}
dS2 = {s: abs(S2[s] - S2['flat']) for s in shapes}

C1 = rng_frozen <= rng_exact + 1e-6
C2 = rng_exact <= 0.01
out = dict(
    E=round(E, 3), budget_frac=0.05,
    Sigma2={s: round(S2[s], 4) for s in shapes},
    damage_exact={s: round(dmg_exact[s], 5) for s in shapes},
    damage_frozenH={s: round(dmg_frozen[s], 5) for s in shapes},
    damage_range_exact=round(rng_exact, 5),
    damage_range_frozenH=round(rng_frozen, 5),
    h_variation_contribution=round(rng_exact - rng_frozen, 5),
    dev_from_flat_exact={s: round(dev_exact[s], 5) for s in shapes},
    dSigma2_from_flat={s: round(dS2[s], 4) for s in shapes},
    C1_frozen_le_exact=bool(C1), C2_invariant=bool(C2),
    note=('frozen-H spread isolates operator-ordering; exact-minus-frozen is the H_t '
          'time-variation (linearization/commutator) error the T1 proof must bound.'),
    runtime_sec=round(time.time() - t0, 1),
)
json.dump(out, open('diag9_commutator_out.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
