"""
Phase-C pilot19 (REAL human annotator noise arm, closes the L1 limitation):

  A (clean)  = CIFAR-10 train subset with CLEAN labels (the ~90.5% of train
               examples whose CIFAR-10N aggre_label == clean_label),
               subsampled to nA for class balance.
  D (conflict)= CIFAR-10 train examples with REAL HUMAN ANNOTATOR DISAGREEMENT
               (aggre_label != clean_label, ~9.5% of train), subsampled to
               dfrac*nA. Labels used = aggre_label (human aggregate) — this is
               genuine per-example human annotation noise, NOT controlled
               corruption (contrast pilot18: cyclic flip = controlled).
  Eval       = CIFAR-10 test (clean).

Data provenance (r210 lane cache, downloaded 2026-08-07):
  CIFAR-10N human labels: data/CIFAR-10_human.pt (Wei et al. 2022, "Learning
  from Noisy Labels with Deep Neural Networks: a Real-world Human Noise Dataset"
  / cifar-10n webside, CC-BY); CIFAR-10 images: data/cifar10 (official tarball).

Arms (mirror pilot17/18): never / always / drop@split / inject@split / inject@late.
Model: SmallCNN on 32x32x3, 10 logits, plain mini-batch SGD, cosine WSD tail
(same hyperparams as pilot17/18, T=240, split=120).

Preregistered (r783):
  R1 object exists with real annotator noise: damage >= 0.01
     (CIFAR-10 CNN acc scale ~0.55-0.75; noise floor far above MNIST)
  R2 erasure-free survives: recovery(drop@split) >= 0.7
  R3 report-only: frac_inject_at_split / frac_inject_late
     (T3 primacy/recency prediction is calibrated on the convex+synthetic
     grid; on real pixels pilot17/18 both show ~0.5-0.9 split, so no floor
     preregistered here — we report the measured value as boundary evidence)
  R4 report-only: acc scale of never arm (sanity vs known CIFAR-10 CNN range)

Method hook (Gate C positive endpoint): if R1&R2 pass with REAL human noise,
"detox window extends to just before LR decay at no recovery cost" holds in
the regime practitioners actually face (few genuinely-mislabeled examples).
"""
import numpy as np, json, time, os, sys
import torch, torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "early_traj_value_r201", "data")

def load_cifar10_images():
    import pickle
    xs, ys = [], []
    for i in range(1, 6):
        with open(os.path.join(DATA, "cifar10", "cifar-10-batches-py", f"data_batch_{i}"), "rb") as f:
            b = pickle.load(f, encoding="bytes")
        xs.append(b[b"data"]); ys.append(np.array(b[b"labels"]))
    X = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.concatenate(ys).astype(np.int64)
    with open(os.path.join(DATA, "cifar10", "cifar-10-batches-py", "test_batch"), "rb") as f:
        b = pickle.load(f, encoding="bytes")
    Xte = b[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    yte = np.array(b[b"labels"]).astype(np.int64)
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)
    return (X - mean) / std, y, (Xte - mean) / std, yte

def load_all(device, seed=0, dfrac=0.10, nA=20000):
    human = torch.load(os.path.join(DATA, "CIFAR-10_human.pt"), map_location="cpu", weights_only=False)
    clean = human["clean_label"]; aggre = human["aggre_label"]
    X, y_clean_file, Xte, yte = load_cifar10_images()
    assert np.array_equal(y_clean_file, clean), "clean_label mismatch with CIFAR-10 tarball labels"
    agree_idx = np.where(aggre == clean)[0]
    disagree_idx = np.where(aggre != clean)[0]
    rng = np.random.default_rng(52_000 + seed)
    nD = int(dfrac * nA)
    selA = rng.choice(agree_idx, size=nA, replace=False)
    selD = rng.choice(disagree_idx, size=min(nD, len(disagree_idx)), replace=False)
    def ten(img, lab):
        return (torch.tensor(img).to(device),
                torch.tensor(lab.astype(np.int64), device=device))
    XA, yA = ten(X[selA], clean[selA])
    XD, yD = ten(X[selD], aggre[selD])  # REAL human aggregate labels
    Xte_t, yte_t = ten(Xte, yte)
    return XA, yA, XD, yD, Xte_t, yte_t, len(disagree_idx) / len(clean)

class SmallCNN(nn.Module):
    def __init__(self, k=32, nclass=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, k, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(k, 2 * k, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * k, 4 * k, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(4 * k * 4 * 4, nclass)
    def forward(self, x): return self.fc(self.net(x).flatten(1))

def eta_of(t, T_split, T, eta_hi, eta_lo):
    if t < T_split: return eta_hi
    ph = (t - T_split) / max(T - T_split, 1)
    return eta_lo + (eta_hi - eta_lo) * 0.5 * (1.0 + np.cos(np.pi * ph))

def train(XA, yA, XD, yD, win, T=240, T_split=120, eta_hi=0.05, eta_lo=0.005, bs=256, seed=0):
    g = torch.Generator(device=DEV).manual_seed(20_000 + seed)
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=eta_hi, momentum=0.0, weight_decay=0.0)
    lossf = nn.CrossEntropyLoss()
    nA, nD = len(yA), len(yD)
    for t in range(T):
        eta = eta_of(t, T_split, T, eta_hi, eta_lo)
        for pg in opt.param_groups: pg["lr"] = eta
        useD = win is not None and win[0] <= t < win[1]
        if useD:
            X = torch.cat([XA, XD]); y = torch.cat([yA, yD]); n = nA + nD
        else:
            X, y, n = XA, yA, nA
        perm = torch.randperm(n, generator=g, device=DEV)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(model(X[idx]), y[idx]); loss.backward(); opt.step()
    return model

@torch.no_grad()
def acc_eval(model, X, y, bs=2000):
    model.eval(); correct = 0
    for i in range(0, len(y), bs):
        correct += (model(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(y)

def main():
    quick = "--quick" in sys.argv
    T, T_split = (12, 6) if quick else (240, 120)
    seeds = [0, 1] if quick else list(range(6))
    dfrac = 0.10; nA = 15000
    t0 = time.time()
    arms = {"never": None, "always": (0, T),
            f"drop@{T_split}": (0, T_split),
            f"inject@{T_split}": (T_split, T),
            "inject@late": (int(0.875 * T), T)}
    rows = {k: [] for k in arms}
    for s in seeds:
        XA, yA, XD, yD, Xte, yte, disagree_frac = load_all(DEV, seed=s, dfrac=dfrac, nA=nA)
        if s == seeds[0]:
            print(f"[data] A clean-agree={len(yA)}  D real-human-disagree={len(yD)} "
                  f"({dfrac:.0%} of A; dataset disagree rate {disagree_frac:.3%})  test={len(yte)} dev={DEV}", flush=True)
        for name, win in arms.items():
            m = train(XA, yA, XD, yD, win, T=T, T_split=T_split, seed=s)
            a = acc_eval(m, Xte, yte)
            rows[name].append(a)
            print(f"  seed{s} {name}: acc={a:.4f} ({time.time()-t0:.0f}s)", flush=True)
    med = lambda v: float(np.median(v))
    acc_never, acc_always = med(rows["never"]), med(rows["always"])
    damage = acc_never - acc_always
    rec_split = (med(rows[f"drop@{T_split}"]) - acc_always) / max(damage, 1e-9)
    frac_split = (med(rows[f"inject@{T_split}"]) - acc_always) / max(damage, 1e-9)
    frac_late = (med(rows["inject@late"]) - acc_always) / max(damage, 1e-9)
    out = dict(
        quick=quick, T=T, seeds=seeds, dev=DEV, nA=nA, nD=len(yD), dfrac=dfrac,
        conflict_block="CIFAR-10N REAL human annotator disagreement (aggre_label!=clean_label), aggre_label used as D label — genuine human noise, NOT controlled corruption",
        data_provenance="CIFAR-10_human.pt (Wei et al. 2022, cached 2026-08-07) + official CIFAR-10 tarball",
        acc={k: round(med(v), 4) for k, v in rows.items()},
        acc_all={k: [round(x, 4) for x in v] for k, v in rows.items()},
        damage=round(damage, 4), recovery_drop_at_split=round(rec_split, 3),
        frac_inject_at_split=round(frac_split, 3), frac_inject_late=round(frac_late, 3),
        R1=bool(damage >= 0.01), R2=bool(rec_split >= 0.7),
        R3_report_only=dict(frac_split=round(frac_split, 3), frac_late=round(frac_late, 3)),
        R4_never_acc=round(acc_never, 4),
        runtime_min=round((time.time() - t0) / 60, 1),
    )
    tag = "quick" if quick else "full"
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, f"pilot19_cifar10n_{tag}_out.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
