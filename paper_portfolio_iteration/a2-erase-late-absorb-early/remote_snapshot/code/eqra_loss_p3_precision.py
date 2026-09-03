"""
P3 precision check: does EQRA-loss collateral-damage a HARMLESS block?

Preregistered (SALVAGE_LOSS_PREREG P3): if D is actually harmless, EQRA-loss acc must be
>= always acc - 0.005 (no collateral). Here D = a CLEAN MNIST subset (labels NOT flipped) —
genuinely harmless. A clean example's loss may still be transiently high early (hard examples),
so this tests the false-positive cost of per-sample-loss quarantine.

Setup mirrors pilot18 (real MNIST pixels, SmallCNN, WSD, 240ep) but D labels are clean.
q_frac=0.10 warmup quarantine + LR-decay re-admission, identical to eqra_loss_salvage.
6 seeds, paired vs always.
"""
import numpy as np, json, time, os, sys
import torch, torch.nn as nn
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p18", os.path.join(HERE, "pilot18_smallconflict.py"))
p18 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p18)
DEV = p18.DEV


def load_clean_block(device, seed=0, dfrac=0.03):
    """Same as pilot18 load_all but D labels are CLEAN (harmless block)."""
    os.makedirs(p18.DROOT, exist_ok=True)
    cache = {}
    for k, u in p18.URLS.items():
        fp = os.path.join(p18.DROOT, k + ".npy")
        cache[k] = np.load(fp)
    def ten(img, lab):
        x = torch.tensor(img.astype(np.float32) / 255.0).unsqueeze(1)
        x = (x - 0.1307) / 0.3081
        return x.to(device), torch.tensor(lab.astype(np.int64), device=device)
    XA, yA = ten(cache["mnist_tr_x"], cache["mnist_tr_y"])
    Xte, yte = ten(cache["mnist_te_x"], cache["mnist_te_y"])
    rng = np.random.default_rng(31_000 + seed)
    nA = len(cache["mnist_tr_y"]); nD = int(dfrac * nA)
    sel = rng.choice(nA, size=nD, replace=False)
    # D = clean subset, labels NOT flipped (harmless)
    XD, yD = ten(cache["mnist_tr_x"][sel], cache["mnist_tr_y"][sel].astype(np.int64))
    return XA, yA, XD, yD, Xte, yte


@torch.no_grad()
def per_sample_loss(model, X, y, bs=4000):
    model.eval()
    out = np.empty(len(y), dtype=np.float32)
    for i in range(0, len(y), bs):
        out[i:i + bs] = nn.functional.cross_entropy(
            model(X[i:i + bs]), y[i:i + bs], reduction="none").cpu().numpy()
    return out


def train_arm(XA, yA, XD, yD, arm, T=240, T_split=120, eta_hi=0.05, eta_lo=0.005,
              bs=256, seed=0, q_frac=0.10):
    g = torch.Generator(device=DEV).manual_seed(50_000 + seed)
    torch.manual_seed(seed); np.random.seed(seed)
    model = p18.SmallCNN().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=eta_hi, momentum=0.0, weight_decay=0.0)
    lossf = nn.CrossEntropyLoss()
    nA, nD = len(yA), len(yD)
    Xall = torch.cat([XA, XD]); yall = torch.cat([yA, yD])
    quar = torch.zeros(nA + nD, dtype=torch.bool, device=DEV)
    for t in range(T):
        eta = p18.eta_of(t, T_split, T, eta_hi, eta_lo)
        for pg in opt.param_groups: pg["lr"] = eta
        if arm == "eqra-loss":
            ls = per_sample_loss(model, Xall, yall)
            if t < T_split:
                k = max(int(q_frac * (nA + nD)), 1)
                thr = np.partition(ls, -k)[-k]
                quar = torch.tensor(ls >= thr, device=DEV)
            else:
                quar = quar & torch.tensor(ls > np.median(ls[:nA]), device=DEV)
        if arm == "never":
            X, y = XA, yA
        elif arm == "always":
            X, y = Xall, yall
        else:
            keep = ~quar; X, y = Xall[keep], yall[keep]
        n = len(y)
        perm = torch.randperm(n, generator=g, device=DEV)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(model(X[idx]), y[idx]); loss.backward(); opt.step()
    return model


def main():
    quick = "--quick" in sys.argv
    T, T_split = (12, 6) if quick else (240, 120)
    seeds = [0, 1] if quick else list(range(6))
    t0 = time.time()
    arms = ["never", "always", "eqra-loss"]
    rows = {k: [] for k in arms}
    for s in seeds:
        XA, yA, XD, yD, Xte, yte = load_clean_block(DEV, seed=s)
        for name in arms:
            m = train_arm(XA, yA, XD, yD, name, T=T, T_split=T_split, seed=s)
            a = p18.acc_mnist(m, Xte, yte)
            rows[name].append(a)
            print(f"  seed{s} {name}: acc={a:.4f} ({time.time()-t0:.0f}s)", flush=True)
    med = lambda v: float(np.median(v))
    acc = {k: med(rows[k]) for k in rows}
    diff = acc["eqra-loss"] - acc["always"]
    out = dict(
        quick=quick, T=T, seeds=seeds,
        setup="D = CLEAN MNIST subset (labels NOT flipped) = HARMLESS block; tests EQRA-loss false-positive cost",
        acc_median={k: round(acc[k], 4) for k in rows},
        acc_all={k: [round(x, 4) for x in rows[k]] for k in rows},
        diff_eqra_vs_always=round(diff, 4),
        P3_pass=bool(diff >= -0.005),
        runtime_min=round((time.time() - t0) / 60, 1),
    )
    tag = "quick" if quick else "full"
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, f"eqra_loss_p3_{tag}_out.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
