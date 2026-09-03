"""pilot16 (r780): does T3's budget traversal transfer to the NONCONVEX MLP cell?
pilot15 proved budget B=K*E_win flips primacy/recency bidirectionally in the convex
cell. Here: same budget-flip design in the MLP+SGD cell (pilot5/7 architecture).

MLP: 40 -> 32 tanh -> 1, mini-batch SGD bs=128, cosine-ish piecewise LR, lam=1e-4.
Window [210,240). Arms vary the in-window LR to change budget:
  B0 base   : eta_lo=0.01 in window (budget x0)   -> expect primacy-lean (lower frac)
  B1 hi     : eta=0.10 in window (10x budget)     -> expect recency-lean (higher frac)
  B2 lo     : eta=0.001 in window (0.1x budget)   -> expect strongest primacy
Same optimizer (SGD bs128) throughout => isolates budget from optimizer identity.

Note: MLP base (pilot5/7) already shows recency (frac~1.0). If T3 transfers, LOWERING
the in-window budget should move frac DOWN toward primacy. That is the falsifiable
direction: budget reduction recovers primacy even in nonconvex.

Preregistered (median over 6 seeds):
  Q1: frac(B2 lo) < frac(B1 hi)            (budget down => primacy-lean)
  Q2: frac(B1 hi) >= frac(B0 base)         (budget up => recency-lean or saturated)
  Q3: monotone frac(B2) <= frac(B0) <= frac(B1)
Report frac = (acc_never - acc_inj@210)/damage.
"""
import numpy as np, json, time, sys
sys.path.insert(0, '.')
from pilot5_boundary import make_data, init_params, forward, loss_grad_batch, metrics

t0 = time.time()
T, T_split, INJ = 240, 120, 210
ETA_HI, ETA_LO, LAM, BS = 0.2, 0.01, 1e-4, 128

def train(XA, yA, XD, yD, win, seed, win_eta):
    rng = np.random.default_rng(10_000 + seed)
    p = init_params(rng)
    for t in range(T):
        # piecewise: hi before split, lo after; override in-window LR
        if t < T_split: eta = ETA_HI
        elif t < INJ: eta = ETA_LO
        else: eta = win_eta
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        perm = rng.permutation(len(y))
        for i in range(0, len(y), BS):
            idx = perm[i:i + BS]
            _, g = loss_grad_batch(p, X[idx], y[idx], LAM)
            for k in p: p[k] = p[k] - eta * g[k]
    return p

arms = {'B0_base': ETA_LO, 'B1_hi': 0.10, 'B2_lo': 0.001}
out = {}
for name, we in arms.items():
    fr = []
    for seed in range(6):
        XA, yA, XD, yD, Xt, yt = make_data(seed)
        p_nv = train(XA, yA, XD, yD, None, seed, we)
        p_al = train(XA, yA, XD, yD, (0, T), seed, we)
        p_ij = train(XA, yA, XD, yD, (INJ, T), seed, we)
        a_nv = metrics(p_nv, XA, yA, Xt, yt)[0]
        a_al = metrics(p_al, XA, yA, Xt, yt)[0]
        a_ij = metrics(p_ij, XA, yA, Xt, yt)[0]
        dmg = a_nv - a_al
        fr.append((a_nv - a_ij) / dmg if dmg > 1e-9 else 0.0)
    out[name] = dict(win_eta=we, frac_med=float(np.median(fr)),
                     frac_all=[round(x, 3) for x in fr])

f = {k: out[k]['frac_med'] for k in out}
Q1 = f['B2_lo'] < f['B1_hi']
Q2 = f['B1_hi'] >= f['B0_base']
Q3 = f['B2_lo'] <= f['B0_base'] <= f['B1_hi']
out['verdicts'] = dict(Q1_budget_down_primacy=bool(Q1), Q2_budget_up_recency=bool(Q2),
                       Q3_monotone=bool(Q3), runtime_sec=round(time.time() - t0, 1))
json.dump(out, open('pilot16_mlp_budget_out.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
