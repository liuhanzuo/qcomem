"""
Phase-1 pilot12 (Gate B tension-2): WHAT in SGD noise drives the primacy->recency flip?

pilot7 (2x2): convex+GD -> primacy (inj@210 frac 0.119); convex+SGD -> recency (1.061).
Confound in pilot7: SGD also does ~10x more parameter updates (10 batches/epoch vs 1).
This pilot holds LR schedule, batch size (128), and update count (2400) EXACTLY fixed;
only the randomness structure changes:

  arm fresh    : standard SGD, fresh permutation each epoch (fresh noise every step)
  arm fixedcyc : data split into 10 fixed batches ONCE (window-independent seed 999);
                 cycled in a fixed order every epoch. Same update count, same batch size,
                 but NO fresh randomness: per-epoch update sequence is deterministic.
  arm fixedmem : batch membership frozen as in fixedcyc, but batch ORDER reshuffled each
                 epoch. Composition noise frozen, order noise fresh. Diagnostic:
                 fixedmem ~ fixedcyc -> composition noise is the driver;
                 fixedmem ~ fresh    -> order noise alone suffices for recency.

Preregistered verdicts (median over 6 seeds, convex cell, same data/schedule as pilot7):
  V1 noise-drives-flip: inj210_frac(fresh) >= 0.7 AND inj210_frac(fixedcyc) <= 0.4
  V2 diagnostic on fixedmem (see above).
"""
import numpy as np, json, time
from pilot5_boundary import make_data
from pilot7_ablate2x2 import sigmoid, T, T_split

ETA_HI, ETA_LO, LAM, BS = 0.8, 0.08, 1e-3, 128


def eta_of_pw(t):
    return ETA_HI if t < T_split else ETA_LO


def make_batches(X, y, ncur, mode, rng):
    """Return list of index arrays for this epoch, per mode. Membership for the two fixed
    modes is built from a window-independent seed so it is identical across windows that
    share composition size."""
    if mode == "fresh":
        perm = rng.permutation(ncur)
        return [perm[i:i + BS] for i in range(0, ncur, BS)]
    base = np.random.default_rng(999).permutation(ncur)  # frozen membership
    batches = [base[i:i + BS] for i in range(0, ncur, BS)]
    if mode == "fixedcyc":
        # exact analogue of `fresh` with the permutation drawn once and REUSED every epoch
        return [np.concatenate(batches)[i:i + BS] for i in range(0, ncur, BS)]
    # fixedmem: frozen membership, fresh order
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def sgd_train(XA, yA, XD, yD, win, seed, mode):
    rng = np.random.default_rng(30_000 + seed)
    w = rng.standard_normal(XA.shape[1]) * 0.05
    for t in range(T):
        eta = eta_of_pw(t)
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        batches = make_batches(X, y, len(y), mode, rng)
        for idx in batches:
            z = X[idx] @ w; p = sigmoid(z)
            g = X[idx].T @ (p - y[idx]) / len(idx) + LAM * w
            w = w - eta * g
    return w


def main():
    t0 = time.time()
    seeds = list(range(6))
    arms = [("never", None), ("always", (0, T)), ("inject@120", (120, T)), ("inject@210", (210, T))]
    modes = ["fresh", "fixedcyc", "fixedmem"]
    out_modes = {}
    for mode in modes:
        acc_rows = {name: [] for name, _ in arms}
        for sd in seeds:
            XA, yA, XD, yD, Xt, yt = make_data(sd)
            for name, win in arms:
                w = sgd_train(XA, yA, XD, yD, win, sd, mode)
                acc_rows[name].append(float(np.mean((Xt @ w > 0) == (yt > 0.5))))
        med = lambda v: float(np.median(v))
        acc_never, acc_always = med(acc_rows["never"]), med(acc_rows["always"])
        damage = acc_never - acc_always
        out_modes[mode] = dict(
            acc={k: round(med(v), 4) for k, v in acc_rows.items()},
            damage=round(damage, 4),
            inj120_frac=round((acc_never - med(acc_rows["inject@120"])) / max(damage, 1e-9), 3),
            inj210_frac=round((acc_never - med(acc_rows["inject@210"])) / max(damage, 1e-9), 3),
        )
    fA = out_modes["fresh"]["inj210_frac"]
    fB = out_modes["fixedcyc"]["inj210_frac"]
    fC = out_modes["fixedmem"]["inj210_frac"]
    V1 = (fA >= 0.7) and (fB <= 0.4)
    diag = "composition-drives" if fC <= 0.4 else ("order-suffices" if fC >= 0.7 else "mixed")
    out = dict(modes=out_modes, V1_noise_drives_flip=bool(V1), fA=fA, fB=fB, fC=fC,
               V2_diagnostic=diag, runtime_sec=round(time.time() - t0, 1))
    print(json.dumps(out, indent=1))
    json.dump(out, open("pilot12_noise_ablate_out.json", "w"), indent=1)


if __name__ == "__main__":
    main()
