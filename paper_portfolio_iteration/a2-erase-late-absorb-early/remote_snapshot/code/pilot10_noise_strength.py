"""
Phase-1 pilot10 (Gate A/C boundary + fix pilot6 P4 NaN): erasure-free law vs CONFLICT
STRENGTH, and loss-side tail probe without the manual-log_loss blowup.

Motivation: pilot1 (separable label noise) had NO object (damage~0); pilot2/5 (full
contradictory near-dup) had a STRONG object (damage~0.34). Where is the conflict-strength
threshold for the object to exist, and does the erasure-free law survive at each level?

Controlled sweep: block D = same features as A but with SYMMETRIC label noise at rate rho
(a fraction rho of D labels flipped). rho=1.0 -> full contradiction (pilot2 regime),
rho=0.5 -> pure noise (max entropy conflict), rho->0 -> clean-consistent (no conflict).
Model: 2-layer MLP, mini-batch SGD, cosine WSD (the realistic pilot5 cell). 6 seeds.

Preregistered:
  E1 object floor: damage(rho) = acc_never - acc_always(rho); report the rho where
     damage first >= 0.05 (object-existence threshold in conflict strength).
  E2 erasure-free per level: acc_rec(drop@120) >= 0.7 for every rho with damage>=0.05.
  E3 LOSS-side tail probe (FIXED): use sklearn log_loss on clipped probabilities
     (p in [1e-7, 1-1e-7]) instead of the manual np.log(sigmoid) that blew up to NaN
     in pilot6 (tail_absf drop@120=NaN). loss_rec(drop@120) >= 0.7 -> erasure is
     function-level real, not a threshold artifact, at each noise level.
  E4 threshold: report weakest rho with full acc+loss recovery (the "erasable object"
     window in conflict strength).
"""
import numpy as np, json, time
from sklearn.metrics import log_loss
from pilot5_boundary import make_data, init_params, forward, loss_grad_batch, apply_grad, \
    eta_of, metrics

T, T_split = 240, 120


def make_noisy_D(XA, yA, rho, seed):
    rng = np.random.default_rng(50_000 + seed)
    XD = XA + 0.05 * rng.standard_normal(XA.shape)
    flip = rng.random(len(yA)) < rho
    yD = np.where(flip, 1.0 - yA, yA)
    return XD, yD


def train_mlp(XA, yA, XD, yD, win, seed, eta_hi=0.2, eta_lo=0.01, lam=1e-4, bs=128):
    rng = np.random.default_rng(10_000 + seed)
    p = init_params(rng)
    for t in range(T):
        eta = eta_of(t, T_split, T, eta_hi, eta_lo)
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        perm = rng.permutation(len(y))
        for i in range(0, len(y), bs):
            idx = perm[i:i + bs]
            _, g = loss_grad_batch(p, X[idx], y[idx], lam)
            apply_grad(p, g, eta)
    return p


def tail_loss(p, Xt, yt):
    """sklearn log_loss on clipped probs -- the fixed pilot6 P4 tail probe."""
    _, f = forward(p, Xt)
    prob = 0.5 * (1.0 + np.tanh(0.5 * f))
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    return float(log_loss(yt, prob, labels=[0.0, 1.0]))


def main():
    t0 = time.time()
    seeds = list(range(6))
    rhos = [1.0, 0.75, 0.5, 0.25, 0.1]
    out_rhos = {}
    for rho in rhos:
        acc = {k: [] for k in ["never", "always", "drop@120"]}
        tl = {k: [] for k in ["never", "always", "drop@120"]}
        for sd in seeds:
            XA, yA, _, _, Xt, yt = make_data(sd)
            XD, yD = make_noisy_D(XA, yA, rho, sd)
            for name, win in [("never", None), ("always", (0, T)), ("drop@120", (0, 120))]:
                p = train_mlp(XA, yA, XD, yD, win, sd)
                a, _, _, _ = metrics(p, XA, yA, Xt, yt)
                acc[name].append(a)
                tl[name].append(tail_loss(p, Xt, yt))
        med = lambda v: float(np.median(v))
        a_nev, a_alw, a_drop = med(acc["never"]), med(acc["always"]), med(acc["drop@120"])
        l_nev, l_alw, l_drop = med(tl["never"]), med(tl["always"]), med(tl["drop@120"])
        dmg = a_nev - a_alw
        acc_rec = (a_drop - a_alw) / max(dmg, 1e-9)
        loss_rec = (l_alw - l_drop) / max(l_alw - l_nev, 1e-9)
        out_rhos[f"rho={rho}"] = dict(
            acc_never=round(a_nev, 4), acc_always=round(a_alw, 4),
            damage=round(dmg, 4), acc_rec_drop120=round(acc_rec, 3),
            loss_never=round(l_nev, 4), loss_always=round(l_alw, 4),
            loss_drop120=round(l_drop, 4), loss_rec_drop120=round(loss_rec, 3),
            object_exists=bool(dmg >= 0.05),
            erasure_acc=bool(acc_rec >= 0.7) if dmg >= 0.05 else None,
            erasure_loss=bool(loss_rec >= 0.7) if dmg >= 0.05 else None,
        )
    # E4: weakest rho with full acc+loss recovery among object-existing levels
    erasable = [r for r in rhos
                if out_rhos[f"rho={r}"]["object_exists"]
                and out_rhos[f"rho={r}"]["erasure_acc"] and out_rhos[f"rho={r}"]["erasure_loss"]]
    out = dict(per_rho=out_rhos,
               weakest_erasable_rho=(min(erasable) if erasable else None),
               E3_probe_fixed=True, runtime_sec=round(time.time() - t0, 1))
    print(json.dumps(out, indent=1))
    json.dump(out, open("pilot10_noise_strength_out.json", "w"), indent=1)


if __name__ == "__main__":
    main()
