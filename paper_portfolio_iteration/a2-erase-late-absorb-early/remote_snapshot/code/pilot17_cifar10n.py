"""
Phase-C pilot17: does EraseLateAbsorbEarly survive REAL-WORLD human label noise?
Data: CIFAR-10 + CIFAR-10N 'worst' labels (Wei et al. 2022, real human annotation
errors, ~40.2% noise rate). Model: small CNN (3 conv blocks + fc). Optimizer:
plain mini-batch SGD (no momentum, no weight decay, no augmentation) to match the
synthetic MLP+SGD cell of pilot5/pilot7/pilot16 where the phenomenon lives.
Schedule: cosine decay eta_hi -> eta_lo over t in [T_split, T) (WSD tail).

Design (mirrors pilot5 arms; D = the noisily-labeled subset):
  - A = clean part: images whose CIFAR-10N 'worse' label AGREES with the CIFAR-10
    true label (these have effectively no injected noise).
  - D = noisy part: images whose 'worst' label DISAGREES with the true label;
    trained with the 'worst' (human-noisy) label.
  - arms: never (A only), always (A+D all T epochs), drop@T_split (D for t<T_split),
          inject@T_split (D only for t>=T_split), inject@210 (D only last 30 ep).
  - eval: CIFAR-10 official TEST set (true labels). damage = acc_never - acc_always.

Preregistered checks (median over seeds):
  C1 object exists on real noise: damage >= 0.02 (2% test acc)
  C2 erasure-free survives: recovery(drop@split) = (acc_drop - acc_always)/damage >= 0.7
  C3 T3 recency/budget prediction: frac = (acc_inj210 - acc_always)/damage >= 0.9
      (MLP+SGD synthetic anchor: inj@210 frac ~1.0, pilot5/pilot7/pilot16)
  C4 T1 invariance qualitatively: mid-window inject@split frac between 0 and 1,
      report only (schedule-shape probe; no floor)
Runtime budget: GPU1, ~6 seeds x 6 arms x 240 ep x ~1.5s/ep ~ 5-8 min.
"""
import numpy as np, json, time, os, sys
import torch, torch.nn as nn, torchvision, torchvision.transforms as T

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = os.path.expanduser("~/.cache/cifar10n")
HERE = os.path.dirname(os.path.abspath(__file__))

def get_data(device):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.247, 0.2435, 0.2616))])
    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR10(DATA, train=False, download=True, transform=tf)
    nl = np.load(os.path.join(DATA, "CIFAR-10_human.pt"), allow_pickle=True)
    worst = nl["worst_label"] if hasattr(nl, "keys") else nl.item()["worst_label"]
    true = np.array(tr.targets)
    # clean part A: where human 'worse' label agrees with true label (no noise)
    agree = np.load(os.path.join(DATA, "CIFAR-10_human.pt"), allow_pickle=True)
    worse = agree["worse_label"] if hasattr(agree, "keys") else agree.item()["worse_label"]
    idxA = np.where(worse == true)[0]
    idxD = np.where(worst != true)[0]
    X = tr.data.astype(np.float32) / 255.0
    def ten(idxs, labels):
        x = torch.tensor(X[idxs]).permute(0, 3, 1, 2)
        m = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        s = torch.tensor([0.247, 0.2435, 0.2616]).view(1, 3, 1, 1)
        x = (x - m) / s
        return x.to(device), torch.tensor(labels, dtype=torch.long, device=device)
    XA, yA = ten(idxA, true[idxA])
    XD, yD = ten(idxD, worst[idxD].astype(np.int64))
    xte = torch.tensor(te.data.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    m = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    s = torch.tensor([0.247, 0.2435, 0.2616]).view(1, 3, 1, 1)
    Xte = ((xte - m) / s).to(device)
    yte = torch.tensor(np.array(te.targets), dtype=torch.long, device=device)
    return XA, yA, XD, yD, Xte, yte

class SmallCNN(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, k, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 16
            nn.Conv2d(k, 2 * k, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 8
            nn.Conv2d(2 * k, 4 * k, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2) # 4
        )
        self.fc = nn.Linear(4 * k * 4 * 4, 10)
    def forward(self, x):
        return self.fc(self.net(x).flatten(1))

def eta_of(t, T_split, T, eta_hi, eta_lo):
    if t < T_split: return eta_hi
    ph = (t - T_split) / max(T - T_split, 1)
    return eta_lo + (eta_hi - eta_lo) * 0.5 * (1.0 + np.cos(np.pi * ph))

def train(XA, yA, XD, yD, win, T=240, T_split=120, eta_hi=0.05, eta_lo=0.005,
          bs=128, seed=0):
    g = torch.Generator(device=DEV).manual_seed(10_000 + seed)
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
            opt.zero_grad()
            loss = lossf(model(X[idx]), y[idx])
            loss.backward(); opt.step()
    return model

@torch.no_grad()
def acc(model, X, y, bs=1000):
    model.eval(); correct = 0
    for i in range(0, len(y), bs):
        correct += (model(X[i:i+bs]).argmax(1) == y[i:i+bs]).sum().item()
    return correct / len(y)

def main():
    quick = "--quick" in sys.argv
    T, T_split = (8, 4) if quick else (240, 120)
    seeds = [0] if quick else list(range(6))
    t0 = time.time()
    XA, yA, XD, yD, Xte, yte = get_data(DEV)
    print(f"[data] A clean={len(yA)}  D noisy={len(yD)}  test={len(yte)}  dev={DEV}",
          flush=True)
    arms = {"never": None, "always": (0, T),
            f"drop@{T_split}": (0, T_split),
            f"inject@{T_split}": (T_split, T),
            "inject@late": (int(0.875 * T), T)}
    rows = {k: [] for k in arms}
    for s in seeds:
        for name, win in arms.items():
            m = train(XA, yA, XD, yD, win, T=T, T_split=T_split, seed=s)
            a = acc(m, Xte, yte)
            rows[name].append(a)
            print(f"  seed{s} {name}: acc={a:.4f} ({time.time()-t0:.0f}s)", flush=True)
    med = lambda v: float(np.median(v))
    acc_never, acc_always = med(rows["never"]), med(rows["always"])
    damage = acc_never - acc_always
    rec_split = (med(rows[f"drop@{T_split}"]) - acc_always) / max(damage, 1e-9)
    frac_split = (med(rows[f"inject@{T_split}"]) - acc_always) / max(damage, 1e-9)
    frac_late = (med(rows["inject@late"]) - acc_always) / max(damage, 1e-9)
    out = dict(
        quick=quick, T=T, seeds=seeds, dev=DEV,
        nA=len(yA), nD=len(yD),
        acc={k: round(med(v), 4) for k, v in rows.items()},
        acc_all={k: [round(x, 4) for x in v] for k, v in rows.items()},
        damage=round(damage, 4),
        recovery_drop_at_split=round(rec_split, 3),
        frac_inject_at_split=round(frac_split, 3),
        frac_inject_late=round(frac_late, 3),
        C1=bool(damage >= 0.02), C2=bool(rec_split >= 0.7), C3=bool(frac_late >= 0.9),
        runtime_min=round((time.time() - t0) / 60, 1),
    )
    tag = "quick" if quick else "full"
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, f"pilot17_cifar10n_{tag}_out.json"), "w"),
              indent=1)

if __name__ == "__main__":
    main()
