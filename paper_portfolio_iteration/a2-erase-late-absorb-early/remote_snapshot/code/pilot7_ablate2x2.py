"""
Phase-1 pilot7 (decisive 2x2 ablation): WHAT drives the primacy->recency flip?

Convex (pilot2/3, full-batch GD): PRIMACY -- damage ~ exposure in high-LR phase,
  inject@210 frac = 0.087.
MLP (pilot5/6, mini-batch SGD): RECENCY -- endpoint set by last ~15-30 steps,
  inject@210 frac = 1.024, drop@235 recovery = 0.726.
Confound: convex pilots used full-batch GD; MLP pilots used mini-batch SGD.
2x2: {convex logistic, MLP} x {full-batch GD, mini-batch SGD}, same data/schedule.

Preregistered (median over 6 seeds):
  Q1 noise-sufficient: cell (convex, SGD) inject@210 frac >= 0.5
  Q2 nonconvexity-sufficient: cell (MLP, GD) inject@210 frac >= 0.5
  Interpretation: Q1&~Q2 -> SGD noise drives recency; ~Q1&Q2 -> nonconvexity;
  both -> either alone; neither -> interaction needed (both required).
  Erasure-law robustness: drop@210 recovery >= 0.7 in ALL four cells.
"""
import numpy as np, json, time
from pilot5_boundary import make_data, init_params, forward, loss_grad_batch, eta_of, metrics

T, T_split = 240, 120


def sigmoid(z): return 0.5 * (1.0 + np.tanh(0.5 * z))


def convex_train(XA, yA, XD, yD, win, seed, sgd, eta_hi=0.8, eta_lo=0.08, lam=1e-3, bs=128):
    rng = np.random.default_rng(20_000 + seed)
    w = rng.standard_normal(XA.shape[1]) * 0.05
    for t in range(T):
        eta = eta_of(t, T_split, T, eta_hi, eta_lo)
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        if sgd:
            perm = rng.permutation(len(y))
            for i in range(0, len(y), bs):
                idx = perm[i:i + bs]
                z = X[idx] @ w; p = sigmoid(z)
                g = X[idx].T @ (p - y[idx]) / len(idx) + lam * w
                w = w - eta * g
        else:
            z = X @ w; p = sigmoid(z)
            g = X.T @ (p - y) / len(y) + lam * w
            w = w - eta * g
    return w


def convex_acc(w, Xt, yt): return float(np.mean((Xt @ w > 0) == (yt > 0.5)))


def mlp_train(XA, yA, XD, yD, win, seed, sgd, eta_hi=0.2, eta_lo=0.01, lam=1e-4, bs=128):
    rng = np.random.default_rng(10_000 + seed)
    p = init_params(rng)
    for t in range(T):
        eta = eta_of(t, T_split, T, eta_hi, eta_lo)
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        if sgd:
            perm = rng.permutation(len(y))
            for i in range(0, len(y), bs):
                idx = perm[i:i + bs]
                _, g = loss_grad_batch(p, X[idx], y[idx], lam)
                for k in p: p[k] = p[k] - eta * g[k]
        else:
            _, g = loss_grad_batch(p, X, y, lam)
            for k in p: p[k] = p[k] - eta * g[k]
    return p


def main():
    t0 = time.time()
    seeds = list(range(6))
    arms = [("never", None), ("always", (0, T)), ("inject@120", (120, T)),
            ("inject@210", (210, T)), ("drop@210", (0, 210)), ("drop@235", (0, 235))]
    cells = [("convex", "GD"), ("convex", "SGD"), ("mlp", "GD"), ("mlp", "SGD")]
    out_cells = {}
    for model, opt in cells:
        acc_rows = {name: [] for name, _ in arms}
        for sd in seeds:
            XA, yA, XD, yD, Xt, yt = make_data(sd)
            for name, win in arms:
                if model == "convex":
                    w = convex_train(XA, yA, XD, yD, win, sd, sgd=(opt == "SGD"))
                    a = convex_acc(w, Xt, yt)
                else:
                    p = mlp_train(XA, yA, XD, yD, win, sd, sgd=(opt == "SGD"))
                    a, _, _, _ = metrics(p, XA, yA, Xt, yt)
                acc_rows[name].append(a)
        med = lambda v: float(np.median(v))
        acc_never, acc_always = med(acc_rows["never"]), med(acc_rows["always"])
        damage = acc_never - acc_always
        cell = dict(
            acc={k: round(med(v), 4) for k, v in acc_rows.items()},
            damage=round(damage, 4),
            inj120_frac=round((acc_never - med(acc_rows["inject@120"])) / max(damage, 1e-9), 3),
            inj210_frac=round((acc_never - med(acc_rows["inject@210"])) / max(damage, 1e-9), 3),
            drop210_rec=round((med(acc_rows["drop@210"]) - acc_always) / max(damage, 1e-9), 3),
            drop235_rec=round((med(acc_rows["drop@235"]) - acc_always) / max(damage, 1e-9), 3),
        )
        out_cells[f"{model}+{opt}"] = cell

    Q1 = out_cells["convex+SGD"]["inj210_frac"] >= 0.5
    Q2 = out_cells["mlp+GD"]["inj210_frac"] >= 0.5
    erase_all = all(c["drop210_rec"] >= 0.7 for c in out_cells.values())
    out = dict(cells=out_cells, Q1_noise_sufficient=bool(Q1),
               Q2_nonconvexity_sufficient=bool(Q2),
               erasure_law_all_cells=bool(erase_all),
               runtime_sec=round(time.time() - t0, 1))
    print(json.dumps(out, indent=1))
    json.dump(out, open("pilot7_ablate2x2_out.json", "w"), indent=1)


if __name__ == "__main__":
    main()
