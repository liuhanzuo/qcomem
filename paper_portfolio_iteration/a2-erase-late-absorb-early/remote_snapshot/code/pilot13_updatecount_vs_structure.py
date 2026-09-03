"""
pilot13 (Gate B Prop3): separate UPDATE-COUNT saturation from WITHIN-EPOCH structure.

r778 analysis: the convex+SGD recency flip is explained by the Prop1 saturation law in
UPDATE count (SGD does K ~ 9.4 updates/epoch, so tail exposure E_update = K*E_epoch
crosses the saturation threshold; 1-exp(-c*E_up) ~ 0.90 at s=210, matching measured
1.061). The MLP GD-vs-SGD gap (0.017 vs 1.031) is NOT explained by update count --
both do 240 epochs vs 2400 updates, yet differ. So within-epoch structure matters in
the nonconvex cell specifically.

This pilot holds UPDATE COUNT and SCHEDULE fixed and varies ONLY whether the K updates
per epoch see (a) the whole data each time (full-batch repeated K times = pure update
count, no within-epoch structure) vs (b) K disjoint mini-batches (within-epoch structure).
If update-count saturation is the whole story, (a) and (b) give the SAME inj@210 frac.
If within-epoch overwrite/leakage adds recency, (b) > (a).

Cell: BOTH convex and MLP (the contrast is the point -- convex should show no (a)-(b)
gap if saturation explains it; MLP should show a gap if within-epoch structure drives it).

Arms per cell (update count = 2400, LR schedule identical, bs chosen so K=9 or 10):
  rep   : each epoch = K full-data gradient steps (pure update count, deterministic)
  mb    : each epoch = K disjoint mini-batch steps (within-epoch structure, standard)
Preregistered verdicts (median over 6 seeds, inj@210 frac):
  V1 convex-saturation: |rep - mb| <= 0.15 in the convex cell (no within-epoch effect)
  V2 mlp-structure:     mb - rep >= 0.2 in the MLP cell (within-epoch adds recency)
"""
import numpy as np, json, time
from pilot5_boundary import make_data, init_params, forward, loss_grad_batch, eta_of
from pilot7_ablate2x2 import sigmoid, convex_train, mlp_train, T, T_split

ETA_HI_C, ETA_LO_C, LAM_C = 0.8, 0.08, 1e-3   # convex cell (pilot7)
ETA_HI_M, ETA_LO_M, LAM_M = 0.2, 0.01, 1e-4   # MLP cell (pilot7)
TOTAL_UPDATES = 2400
N_EPOCHS = T  # 240 epochs -> K = TOTAL_UPDATES / T = 10 updates per epoch


def convex_rep(XA, yA, XD, yD, win, seed):
    """K full-data gradient steps per epoch (pure update count)."""
    rng = np.random.default_rng(50_000 + seed)
    w = rng.standard_normal(XA.shape[1]) * 0.05
    K = TOTAL_UPDATES // N_EPOCHS
    for t in range(N_EPOCHS):
        eta = ETA_HI_C if t < T_split else ETA_LO_C
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        for _ in range(K):
            z = X @ w; p = sigmoid(z)
            g = X.T @ (p - y) / len(y) + LAM_C * w
            w = w - eta * g
    return w


def convex_mb(XA, yA, XD, yD, win, seed):
    """K disjoint mini-batch steps per epoch (within-epoch structure)."""
    rng = np.random.default_rng(50_000 + seed)
    w = rng.standard_normal(XA.shape[1]) * 0.05
    K = TOTAL_UPDATES // N_EPOCHS
    for t in range(N_EPOCHS):
        eta = ETA_HI_C if t < T_split else ETA_LO_C
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        bs = max(len(y) // K, 1)
        perm = rng.permutation(len(y))
        for i in range(0, len(y), bs):
            idx = perm[i:i + bs]
            z = X[idx] @ w; p = sigmoid(z)
            g = X[idx].T @ (p - y[idx]) / len(idx) + LAM_C * w
            w = w - eta * g
    return w


def mlp_rep(XA, yA, XD, yD, win, seed):
    rng = np.random.default_rng(60_000 + seed)
    p = init_params(rng)
    K = TOTAL_UPDATES // N_EPOCHS
    for t in range(N_EPOCHS):
        eta = ETA_HI_M if t < T_split else ETA_LO_M
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        for _ in range(K):
            _, g = loss_grad_batch(p, X, y, LAM_M)
            for k in p: p[k] = p[k] - eta * g[k]
    return p


def mlp_mb(XA, yA, XD, yD, win, seed):
    rng = np.random.default_rng(60_000 + seed)
    p = init_params(rng)
    K = TOTAL_UPDATES // N_EPOCHS
    for t in range(N_EPOCHS):
        eta = ETA_HI_M if t < T_split else ETA_LO_M
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        bs = max(len(y) // K, 1)
        perm = rng.permutation(len(y))
        for i in range(0, len(y), bs):
            idx = perm[i:i + bs]
            _, g = loss_grad_batch(p, X[idx], y[idx], LAM_M)
            for k in p: p[k] = p[k] - eta * g[k]
    return p


def acc_of(w_or_p, Xt, yt, cell):
    if cell == "convex":
        return float(np.mean((Xt @ w_or_p > 0) == (yt > 0.5)))
    _, f = forward(w_or_p, Xt)
    return float(np.mean((f > 0) == (yt > 0.5)))


def run_cell(cell, rep_fn, mb_fn, seeds):
    arms = [("never", None), ("always", (0, T)), ("inject@210", (210, T))]
    res = {}
    for mode, fn in [("rep", rep_fn), ("mb", mb_fn)]:
        rows = {n: [] for n, _ in arms}
        for sd in seeds:
            XA, yA, XD, yD, Xt, yt = make_data(sd)
            for name, win in arms:
                w = fn(XA, yA, XD, yD, win, sd)
                rows[name].append(acc_of(w, Xt, yt, cell))
        med = lambda v: float(np.median(v))
        an, aa = med(rows["never"]), med(rows["always"])
        dmg = an - aa
        res[mode] = dict(acc={k: round(med(v), 4) for k, v in rows.items()},
                         damage=round(dmg, 4),
                         inj210_frac=round((an - med(rows["inject@210"])) / max(dmg, 1e-9), 3))
    return res


def main():
    t0 = time.time()
    seeds = list(range(6))
    convex = run_cell("convex", convex_rep, convex_mb, seeds)
    mlp = run_cell("mlp", mlp_rep, mlp_mb, seeds)
    gap_convex = convex["mb"]["inj210_frac"] - convex["rep"]["inj210_frac"]
    gap_mlp = mlp["mb"]["inj210_frac"] - mlp["rep"]["inj210_frac"]
    V1 = abs(gap_convex) <= 0.15
    V2 = gap_mlp >= 0.2
    out = dict(convex=convex, mlp=mlp,
               gap_convex=round(gap_convex, 3), gap_mlp=round(gap_mlp, 3),
               V1_convex_saturation=bool(V1), V2_mlp_structure=bool(V2),
               runtime_sec=round(time.time() - t0, 1))
    print(json.dumps(out, indent=1))
    json.dump(out, open("pilot13_updatecount_vs_structure_out.json", "w"), indent=1)


if __name__ == "__main__":
    main()
