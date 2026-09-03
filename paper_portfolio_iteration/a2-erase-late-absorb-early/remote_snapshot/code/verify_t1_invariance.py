"""verify_t1_invariance (r780): probe WHY always-on damage is per-seed EXACTLY equal
across schedule shapes at fixed E=sum(eta_t) (diag_seam: base/flat/mild identical).

Hypothesis (T1 lemma): in the always-on arm, w converges to the SAME joint attractor
w*_{A∪D} regardless of schedule shape, because strong convexity gives a unique
minimizer and GD converges to it given enough total budget. Schedule shape only
changes the TRANSIENT, not the fixed point. Hence the terminal w_T (and damage) is
schedule-independent — provided every schedule reaches the attractor (saturated).

Contrast with the TRANSIENT regime (inject window, pilot15): there budget is
insufficient, so coverage (and damage) DOES depend on K*E_win — schedule matters.

This script tests the mechanism-level claim:
  H1: terminal distance ||w_T - w*_{A∪D}|| is ~0 for all three schedule shapes
      (base/flat/mild) at fixed E  => all converge to the same fixed point.
  H2: the per-seed damage equality follows from H1 (same fixed point => same acc).
  H3 (boundary/falsifier): if we STARVE the budget (tiny E), the three schedules
      should NO LONGER all reach the attractor, and damage should become
      schedule-DEPENDENT (transient regime) => invariance breaks. This is the
      transient/saturation boundary = the T1<->T3 seam, made quantitative.

Setup: convex ridge logistic, always-on D, 3 schedule shapes at fixed E, plus a
starved-E control (E scaled by 0.05) for H3. 6 seeds.
"""
import numpy as np, json, time, sys
sys.path.insert(0, '.')
from pilot7_ablate2x2 import sigmoid
from pilot5_boundary import make_data

t0 = time.time()
lam = 1e-3
T, T_split = 240, 120

def train_eta(XA, yA, XD, yD, ETA, w0, always=True):
    w = w0.copy()
    for t in range(len(ETA)):
        blocks = [(XA, yA)] + ([(XD, yD)] if always else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        z = X @ w; p = sigmoid(z); g = X.T @ (p - y) / len(y) + lam * w
        w = w - ETA[t] * g
    return w

def acc(w, Xt, yt): return float(np.mean((Xt @ w > 0) == (yt > 0.5)))

def sched(shape, E, T=240, split=120):
    """piecewise-constant schedule with total exposure E, given hi/lo shape."""
    if shape == 'base':  # 0.8/0.08 reference scaled to E
        eh, el = 0.8, 0.08
    elif shape == 'flat':
        eh = el = None
    elif shape == 'mild':
        eh, el = 0.6, 0.28
    if shape == 'flat':
        v = E / T
        return np.full(T, v)
    scale = E / (eh * split + el * (T - split))
    return np.array([eh * scale if t < split else el * scale for t in range(T)])

E_full = 0.8 * 120 + 0.08 * 120     # 105.6
E_starve = E_full * 0.05            # starved budget for H3

shapes = ['base', 'flat', 'mild']
res = {}
for regime, E in [('fullE', E_full), ('starvedE', E_starve)]:
    rows = {}
    dmg_by_shape = {s: [] for s in shapes}
    dist_by_shape = {s: [] for s in shapes}
    for seed in range(6):
        XA, yA, XD, yD, Xt, yt = make_data(seed)
        w0 = np.random.default_rng(1000 + seed).standard_normal(40) * 0.05
        # clean attractor w*_A (long deterministic GD on A only, big budget)
        w_starA = train_eta(XA, yA, XD, yD, np.full(400, 0.3), w0, always=False)
        # joint attractor w*_{A∪D} (long deterministic GD on A+D)
        w_starAD = train_eta(XA, yA, XD, yD, np.full(400, 0.3), w0, always=True)
        a_nv = acc(w_starA, Xt, yt)
        for s in shapes:
            ETA = sched(s, E)
            w_al = train_eta(XA, yA, XD, yD, ETA, w0, always=True)
            dist = float(np.linalg.norm(w_al - w_starAD))   # distance to joint attractor
            dmg = a_nv - acc(w_al, Xt, yt)
            dist_by_shape[s].append(dist)
            dmg_by_shape[s].append(dmg)
    for s in shapes:
        rows[s] = dict(dist_to_attractor_med=float(np.median(dist_by_shape[s])),
                       dist_max=float(np.max(dist_by_shape[s])),
                       damage_med=float(np.median(dmg_by_shape[s])),
                       damage_per_seed=[round(x, 4) for x in dmg_by_shape[s]])
    dmgs = np.array([np.median(dmg_by_shape[s]) for s in shapes])
    dists = np.array([np.median(dist_by_shape[s]) for s in shapes])
    rows['_spread'] = dict(damage_range=float(dmgs.max() - dmgs.min()),
                           dist_max_over_shapes=float(dists.max()))
    res[regime] = rows

# verdicts
H1 = all(res['fullE'][s]['dist_to_attractor_med'] < 0.05 for s in shapes)
H2 = res['fullE']['_spread']['damage_range'] < 0.01
# H3: starved budget -> at least one shape fails to converge (dist large) AND
#     damage spread grows vs fullE
starve_dists = [res['starvedE'][s]['dist_to_attractor_med'] for s in shapes]
H3_break = (max(starve_dists) > 0.3) and (res['starvedE']['_spread']['damage_range'] >
                                          res['fullE']['_spread']['damage_range'])
res['verdicts'] = dict(H1_fullE_converge_same_attractor=bool(H1),
                       H2_damage_schedule_invariant=bool(H2),
                       H3_starvedE_invariance_breaks=bool(H3_break),
                       fullE_dist=dists.tolist(), starvedE_dist=starve_dists,
                       runtime_sec=round(time.time() - t0, 1))
json.dump(res, open('verify_t1_invariance_out.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
