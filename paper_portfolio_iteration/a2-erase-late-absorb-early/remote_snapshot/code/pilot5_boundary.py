"""
Phase-1 pilot5 (boundary map): does the EraseLateAbsorbEarly phenomenon survive
NONCONVEX + MINI-BATCH + COSINE-DECAY WSD training?

Change vs pilot2/3 (convex ridge logistic, full-batch GD, piecewise-constant LR):
  - model: 2-layer MLP (d=40 -> h=32 tanh -> 1), plain mini-batch SGD (no momentum;
    momentum is a separate boundary question -> pilot6 if needed)
  - LR: cosine decay from eta_hi to eta_lo over t in [T_split, T)
  - data: same Gaussian-mixture + contradictory block D (near-dup features, flipped labels)

Preregistered floors (median over 8 seeds):
  N1 object exists: damage = acc_never - acc_always >= 0.05
  N2 erasure-free survives: recovery(drop@T_split) = (acc_drop - acc_always)/damage >= 0.7
  N3 late-drop still recovers: recovery(drop@0.875T) >= 0.5
  N4 FUNCTION-LEVEL erasure (threshold-artifact falsifier):
      loss-side recovery = (loss_always - loss_drop)/(loss_always - loss_never) >= 0.7
      (if acc recovers but calibrated loss does not, "erasure" is a threshold/scale artifact)
  N5 mechanism signature (report only): ||theta||_always > ||theta||_never at end,
      and mean signed margin on clean train: margin_never >= margin_always
      (conflict = norm inflation / margin compression; erasure frees capacity)
"""
import numpy as np, json, time


def make_data(seed, d=40, nA=600, nt=4000, sep=2.2, eps=0.05):
    rng = np.random.default_rng(seed)
    mu = np.zeros(d); mu[0] = 1.0
    XA = rng.standard_normal((nA, d)); yA = (rng.random(nA) < 0.5).astype(float)
    XA += (2 * yA[:, None] - 1) * (sep / 2) * mu
    XD = XA + eps * rng.standard_normal((nA, d)); yD = 1.0 - yA
    Xt = rng.standard_normal((nt, d)); yt = (rng.random(nt) < 0.5).astype(float)
    Xt += (2 * yt[:, None] - 1) * (sep / 2) * mu
    return XA, yA, XD, yD, Xt, yt


def init_params(rng, d=40, h=32):
    return dict(W1=rng.standard_normal((h, d)) / np.sqrt(d),
                b1=np.zeros(h),
                w2=rng.standard_normal(h) / np.sqrt(h),
                b2=0.0)


def forward(p, X):
    H = np.tanh(X @ p["W1"].T + p["b1"])          # (n,h)
    f = H @ p["w2"] + p["b2"]                     # (n,)
    return H, f


def loss_grad_batch(p, X, y, lam):
    n = len(y)
    H, f = forward(p, X)
    loss = float(np.mean(np.logaddexp(0, f) - y * f))
    z = (0.5 * (1.0 + np.tanh(0.5 * f)) - y) / n  # d loss / d f
    g_w2 = H.T @ z + lam * p["w2"]
    g_b2 = float(z.sum())
    dh = np.outer(z, p["w2"]) * (1.0 - H ** 2)    # (n,h)
    g_W1 = dh.T @ X + lam * p["W1"]
    g_b1 = dh.sum(0)
    reg = 0.5 * lam * (p["W1"] ** 2).sum() + 0.5 * lam * (p["w2"] ** 2).sum()
    return loss + float(reg), dict(W1=g_W1, b1=g_b1, w2=g_w2, b2=g_b2)


def apply_grad(p, g, eta):
    for k in p: p[k] = p[k] - eta * g[k]


def eta_of(t, T_split, T, eta_hi, eta_lo):
    if t < T_split: return eta_hi
    ph = (t - T_split) / max(T - T_split, 1)
    return eta_lo + (eta_hi - eta_lo) * 0.5 * (1.0 + np.cos(np.pi * ph))


def train(XA, yA, XD, yD, win, T=240, T_split=120, eta_hi=0.2, eta_lo=0.01,
          lam=1e-4, bs=128, seed=0, snap_at=()):
    rng = np.random.default_rng(10_000 + seed)
    p = init_params(rng)
    snaps = {}
    for t in range(T):
        eta = eta_of(t, T_split, T, eta_hi, eta_lo)
        blocks = [(XA, yA)]
        if win is not None and win[0] <= t < win[1]:
            blocks.append((XD, yD))
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        perm = rng.permutation(len(y))
        for i in range(0, len(y), bs):
            idx = perm[i:i + bs]
            _, g = loss_grad_batch(p, X[idx], y[idx], lam)
            apply_grad(p, g, eta)
        if t in snap_at:
            snaps[t] = dict(norm=float(np.sqrt((p["W1"] ** 2).sum() + (p["w2"] ** 2).sum())))
    return p, snaps


def metrics(p, XA, yA, Xt, yt):
    _, ft = forward(p, Xt)
    acc = float(np.mean((ft > 0) == (yt > 0.5)))
    loss = float(np.mean(np.logaddexp(0, ft) - yt * ft))
    _, fa = forward(p, XA)
    margin = float(np.mean((2 * yA - 1) * fa))
    norm = float(np.sqrt((p["W1"] ** 2).sum() + (p["w2"] ** 2).sum()))
    return acc, loss, margin, norm


def main():
    t0 = time.time()
    T, T_split = 240, 120
    drop_late = int(0.875 * T)  # 210
    seeds = list(range(8))
    arms = {
        "never": None,
        "always": (0, T),
        f"drop@{T_split}": (0, T_split),
        f"drop@{drop_late}": (0, drop_late),
        f"inject@{T_split}": (T_split, T),
        f"inject@{drop_late}": (drop_late, T),
    }
    rows = {k: [] for k in arms}
    norms_end = {k: [] for k in ["never", "always", f"drop@{T_split}"]}
    margins_end = {k: [] for k in ["never", "always", f"drop@{T_split}"]}
    losses_end = {k: [] for k in arms}
    for s in seeds:
        XA, yA, XD, yD, Xt, yt = make_data(s)
        for name, win in arms.items():
            p, _ = train(XA, yA, XD, yD, win, T=T, T_split=T_split, seed=s)
            acc, loss, margin, norm = metrics(p, XA, yA, Xt, yt)
            rows[name].append(acc); losses_end[name].append(loss)
            if name in norms_end:
                norms_end[name].append(norm); margins_end[name].append(margin)

    med = lambda v: float(np.median(v))
    acc_never, acc_always = med(rows["never"]), med(rows["always"])
    loss_never, loss_always = med(losses_end["never"]), med(losses_end["always"])
    damage = acc_never - acc_always
    k_split, k_late = f"drop@{T_split}", f"drop@{drop_late}"
    rec_split = (med(rows[k_split]) - acc_always) / max(damage, 1e-9)
    rec_late = (med(rows[k_late]) - acc_always) / max(damage, 1e-9)
    lrec_split = ((loss_always - med(losses_end[k_split]))
                  / max(loss_always - loss_never, 1e-9))
    out = dict(
        acc={k: round(med(v), 4) for k, v in rows.items()},
        loss={k: round(med(v), 4) for k, v in losses_end.items()},
        damage=round(damage, 4),
        recovery_drop_at_split=round(rec_split, 3),
        recovery_drop_late=round(rec_late, 3),
        loss_recovery_drop_at_split=round(lrec_split, 3),
        margin_end={k: round(med(v), 3) for k, v in margins_end.items()},
        norm_end={k: round(med(v), 3) for k, v in norms_end.items()},
        N1=bool(damage >= 0.05), N2=bool(rec_split >= 0.7),
        N3=bool(rec_late >= 0.5), N4=bool(lrec_split >= 0.7),
        N5a=bool(med(norms_end["always"]) > med(norms_end["never"])),
        N5b=bool(med(margins_end["never"]) >= med(margins_end["always"])),
        runtime_sec=round(time.time() - t0, 1),
    )
    print(json.dumps(out, indent=1))
    json.dump(out, open("pilot5_boundary_out.json", "w"), indent=1)


if __name__ == "__main__":
    main()
