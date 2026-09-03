"""
Phase-0 pilot (object-existence): LR-phase-dependent data value.

Setup: 2-class Gaussian mixture, ridge logistic regression, full-batch GD with
piecewise-constant LR: eta_hi for t in [0, T_split), eta_lo for [T_split, T].
Two data blocks: A (clean, aligned with class means), D (corrupted-label block).

Preregistered hypotheses (effect floor):
  H1: early-drop (train without D in phase 1 only) recovers >= 60% of the
      test-accuracy damage of always-D, relative to never-D oracle.
  H2: late-drop recovers <= 40%.
  H3: monotonicity of drop-time benefit is NOT flat (Kendall tau between
      drop-time and recovery >= 0.7 across >= 6 drop times).

Also measure swap direction: train clean first, inject D late (poison-late).
"""
import numpy as np, json, time

def sigmoid(z): return 0.5 * (1.0 + np.tanh(0.5 * z))

def make_data(n_per_class=400, d=40, sep=2.2, seed=0, label_noise=0.0, rng=None):
    rng = np.random.default_rng(seed) if rng is None else rng
    mu = np.zeros(d); mu[0] = 1.0
    X = rng.standard_normal((2 * n_per_class, d))
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    X += (2 * y[:, None] - 1.0) * (sep / 2.0) * mu
    if label_noise > 0:
        flip = rng.random(2 * n_per_class) < label_noise
        y = y.copy(); y[flip] = 1 - y[flip]
    return X, y.astype(np.float64)

def loss_grad(w, X, y, lam):
    z = X @ w; p = sigmoid(z)
    n = len(y)
    loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w @ w
    g = X.T @ (p - y) / n + lam * w
    return loss, g

def train(blocks_schedule, T, eta_hi, eta_lo, T_split, lam, w0):
    """blocks_schedule: list of (start,end,X,y) segments active."""
    w = w0.copy()
    for t in range(T):
        eta = eta_hi if t < T_split else eta_lo
        # accumulate grad over active blocks
        gsum = np.zeros_like(w); ntot = 0
        for (s, e, X, y) in blocks_schedule:
            if s <= t < e:
                _, g = loss_grad(w, X, y, 0.0)
                gsum += g * len(y); ntot += len(y)
        g = gsum / max(ntot, 1) + lam * w
        w = w - eta * g
    return w

def acc(w, X, y):
    return float(np.mean((X @ w > 0) == (y > 0.5)))

def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    d = 40; T = 240; T_split = 120
    eta_hi, eta_lo = 0.8, 0.08   # 10x decay at midpoint (WSD-like)
    lam = 1e-3
    # clean block A and corrupted block D (30% label noise), equal size
    XA, yA = make_data(n_per_class=300, d=d, sep=2.2, seed=1)
    XD, yD = make_data(n_per_class=300, d=d, sep=2.2, seed=2, label_noise=0.35)
    Xt, yt = make_data(n_per_class=2000, d=d, sep=2.2, seed=99)  # clean test

    w0 = rng.standard_normal(d) * 0.05

    def sched(include_D, D_window=None):
        segs = [(0, T, XA, yA)]
        if include_D:
            s, e = D_window
            segs.append((s, e, XD, yD))
        return segs

    acc_never = acc(train(sched(False), T, eta_hi, eta_lo, T_split, lam, w0), Xt, yt)
    acc_always = acc(train(sched(True, (0, T)), T, eta_hi, eta_lo, T_split, lam, w0), Xt, yt)

    # drop-time sweep: D present [0, s) only, dropped at s
    drop_times = [0, 30, 60, 90, 120, 150, 180, 210, 240]
    recs = []
    damage = acc_never - acc_always
    for s in drop_times:
        a = acc(train(sched(True, (0, s)), T, eta_hi, eta_lo, T_split, lam, w0), Xt, yt)
        rec = 1.0 if s == 0 else (a - acc_always) / max(damage, 1e-12)
        recs.append((s, a, rec))

    # inject-time sweep: D present [s, T) only (poison at s)
    inject = []
    for s in drop_times[1:]:
        a = acc(train(sched(True, (s, T)), T, eta_hi, eta_lo, T_split, lam, w0), Xt, yt)
        dmg_inj = acc_never - a
        inject.append((s, a, dmg_inj / max(damage, 1e-12)))

    # H1/H2 readout
    rec_early = next(r for (s, _, r) in recs if s == T_split)   # drop exactly at decay
    rec_late = next(r for (s, _, r) in recs if s == 210)
    # Kendall tau between drop time and recovery
    taus = [r for (_, _, r) in recs]
    n = len(drop_times); conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            di = drop_times[j] - drop_times[i]; ri = taus[j] - taus[i]
            if di * ri > 0: conc += 1
            elif di * ri < 0: disc += 1
    kendall = (conc - disc) / (conc + disc + 1e-12)

    out = dict(
        acc_never=acc_never, acc_always=acc_always, damage=damage,
        drop_curve=[(s, round(a, 4), round(r, 4)) for (s, a, r) in recs],
        inject_curve=[(s, round(a, 4), round(r, 4)) for (s, a, r) in inject],
        rec_drop_at_decay=rec_early, rec_drop_late=rec_late,
        kendall_drop_recovery=kendall,
        H1_pass=bool(rec_early >= 0.60), H2_pass=bool(rec_late <= 0.40),
        H3_pass=bool(kendall >= 0.7),
        runtime_sec=time.time() - t0,
        config=dict(T=T, T_split=T_split, eta_hi=eta_hi, eta_lo=eta_lo, lam=lam,
                    nA=600, nD=600, d=d, noise_D=0.35),
    )
    print(json.dumps(out, indent=1))
    with open("pilot_phase_datavalue_out.json", "w") as f:
        json.dump(out, f, indent=1)

if __name__ == "__main__":
    main()
