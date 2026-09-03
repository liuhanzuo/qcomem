"""
Phase-0 pilot v2 (object-existence, gradient-conflict regime).

Mechanism hypothesis: label noise on a SEPARABLE cluster gets fit (memorized)
with no conflict -> no damage. Real damage requires gradient conflict between
blocks. Construction: D = near-duplicate features of A with FLIPPED labels
(contradictory supervision) => strong persistent gradient conflict.

Preregistered (floors):
  H1: rec(drop at LR-decay boundary t=Tsplit) >= 0.5
  H2: rec(drop late, t=0.875T) <= 0.35
  H3: Kendall tau(drop_time, recovery) >= 0.7 over >= 6 drop times
  H4 (asymmetry): inject-late damage fraction >= 0.5 of always-D damage
"""
import numpy as np, json, time

def sigmoid(z): return 0.5 * (1.0 + np.tanh(0.5 * z))

def loss_grad(w, X, y, lam):
    z = X @ w; p = sigmoid(z)
    loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w @ w
    g = X.T @ (p - y) / len(y) + lam * w
    return loss, g

def train(segs, T, eta_hi, eta_lo, T_split, lam, w0):
    w = w0.copy()
    for t in range(T):
        eta = eta_hi if t < T_split else eta_lo
        gsum = np.zeros_like(w); ntot = 0
        for (s, e, X, y) in segs:
            if s <= t < e:
                _, g = loss_grad(w, X, y, 0.0)
                gsum += g * len(y); ntot += len(y)
        g = gsum / max(ntot, 1) + lam * w
        w = w - eta * g
    return w

def acc(w, X, y): return float(np.mean((X @ w > 0) == (y > 0.5)))

def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    d = 40; T = 240; T_split = 120
    eta_hi, eta_lo = 0.8, 0.08
    lam = 1e-3
    nA = 600
    mu = np.zeros(d); mu[0] = 1.0
    sep = 2.2
    XA = rng.standard_normal((nA, d)); yA = (rng.random(nA) < 0.5).astype(float)
    XA += (2 * yA[:, None] - 1) * (sep / 2) * mu
    eps = 0.05
    XD = XA + eps * rng.standard_normal((nA, d))
    yD = 1.0 - yA
    nt = 4000
    Xt = rng.standard_normal((nt, d)); yt = (rng.random(nt) < 0.5).astype(float)
    Xt += (2 * yt[:, None] - 1) * (sep / 2) * mu

    w0 = rng.standard_normal(d) * 0.05

    def sched(include_D, win=None):
        segs = [(0, T, XA, yA)]
        if include_D:
            segs.append((win[0], win[1], XD, yD))
        return segs

    w_never = train(sched(False), T, eta_hi, eta_lo, T_split, lam, w0)
    w_always = train(sched(True, (0, T)), T, eta_hi, eta_lo, T_split, lam, w0)
    acc_never, acc_always = acc(w_never, Xt, yt), acc(w_always, Xt, yt)
    damage = acc_never - acc_always

    drop_times = [0, 30, 60, 90, 120, 150, 180, 210, 240]
    recs = []
    for s in drop_times:
        w = train(sched(True, (0, s)), T, eta_hi, eta_lo, T_split, lam, w0)
        a = acc(w, Xt, yt)
        rec = 1.0 if s == 0 else (a - acc_always) / damage
        recs.append((s, a, rec))

    inject = []
    for s in drop_times[1:]:
        w = train(sched(True, (s, T)), T, eta_hi, eta_lo, T_split, lam, w0)
        a = acc(w, Xt, yt)
        inject.append((s, a, (acc_never - a) / damage))

    rec_at_decay = next(r for (s, _, r) in recs if s == T_split)
    rec_late = next(r for (s, _, r) in recs if s == 210)
    taus = [r for (_, _, r) in recs]
    conc = disc = 0
    for i in range(len(drop_times)):
        for j in range(i + 1, len(drop_times)):
            prod = (drop_times[j] - drop_times[i]) * (taus[j] - taus[i])
            conc += prod > 0; disc += prod < 0
    kendall = (conc - disc) / (conc + disc + 1e-12)
    inj_late = next(r for (s, _, r) in inject if s == 210)

    out = dict(
        acc_never=round(acc_never, 4), acc_always=round(acc_always, 4),
        damage=round(damage, 4),
        drop_curve=[(s, round(a, 4), round(r, 3)) for (s, a, r) in recs],
        inject_curve=[(s, round(a, 4), round(r, 3)) for (s, a, r) in inject],
        rec_drop_at_decay=round(rec_at_decay, 3), rec_drop_late=round(rec_late, 3),
        inject_late_frac=round(inj_late, 3),
        kendall=round(kendall, 3),
        H1=bool(rec_at_decay >= 0.5), H2=bool(rec_late <= 0.35),
        H3=bool(kendall >= 0.7), H4=bool(inj_late >= 0.5),
        runtime_sec=round(time.time() - t0, 2),
    )
    print(json.dumps(out, indent=1))
    json.dump(out, open("pilot2_conflict_out.json", "w"), indent=1)

if __name__ == "__main__":
    main()
