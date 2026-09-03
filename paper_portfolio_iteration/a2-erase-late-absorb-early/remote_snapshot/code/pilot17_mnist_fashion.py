"""
Phase-C pilot17 (REAL-DATA arm): does EraseLateAbsorbEarly survive a real
distribution-conflict block instead of synthetic Gaussian contradictory points?

  A (clean)  = MNIST train (0-9), eval on MNIST test.
  D (conflict)= REAL MNIST train images with SHUFFLED labels (a fixed random
                permutation of A's own labels). Same input manifold as A (real
                pixels, not synthetic Gaussian) but an OPPOSED labeling function
                -> genuine gradient conflict on the same 10-way task, the real-data
                analog of the synthetic near-duplicate flipped-label block D.
                (First attempt used Fashion-MNIST as D under fresh logits: damage
                was ~1e-3, all arms within 0.0009 -- a disjoint-labeling off-task
                block produces NO meaningful conflict with the clean task. Shuffled
                labels on the SAME manifold is the correct conflict construction.)

Model: small CNN (2 conv blocks + fc), 10 output logits. Plain mini-batch SGD
(no momentum/wd/aug) to match the synthetic MLP+SGD cell where recency/erasure
lives. Cosine WSD tail.

Arms (mirror pilot5): never / always / drop@split / inject@split / inject@late.
damage = acc_never - acc_always on MNIST test.

Preregistered (median over seeds):
  R1 object exists with real conflict block: damage >= 0.01 (1% MNIST acc)
  R2 erasure-free survives: recovery(drop@split)=(acc_drop-acc_always)/damage >= 0.7
  R3 T3 recency/budget: frac_late=(acc_injLate-acc_always)/damage >= 0.9
      (synthetic SGD anchor ~1.0; real-conflict replication)
  R4 report-only: frac_split in (0,1) partial recency gradient
Runtime: GPU1, 6 seeds x 5 arms x 240 ep; ~64k imgs/epoch small CNN ~ few min/arm.
"""
import numpy as np, json, time, os, sys, io
import torch, torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
DROOT = os.path.expanduser("~/.cache/mnist_fashion")

URLS = {
    "mnist_tr_x": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "mnist_tr_y": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "mnist_te_x": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "mnist_te_y": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    "fash_tr_x": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    "fash_tr_y": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
}

def _fetch(url):
    import urllib.request, gzip
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    return gzip.decompress(raw)

def _idx_images(b):
    n = int.from_bytes(b[4:8], "big"); r = int.from_bytes(b[8:12], "big"); c = int.from_bytes(b[12:16], "big")
    return np.frombuffer(b[16:], np.uint8).reshape(n, r, c)

def _idx_labels(b):
    n = int.from_bytes(b[4:8], "big"); return np.frombuffer(b[8:8 + n], np.uint8)

def load_all(device, seed=0):
    os.makedirs(DROOT, exist_ok=True)
    cache = {}
    for k, u in URLS.items():
        fp = os.path.join(DROOT, k + ".npy")
        if os.path.exists(fp):
            cache[k] = np.load(fp)
        else:
            b = _fetch(u)
            arr = _idx_images(b) if "_x" in k else _idx_labels(b)
            np.save(fp, arr); cache[k] = arr
    def ten(img, lab):
        x = torch.tensor(img.astype(np.float32) / 255.0).unsqueeze(1)  # (n,1,28,28)
        x = (x - 0.1307) / 0.3081
        return x.to(device), torch.tensor(lab.astype(np.int64), device=device)
    XA, yA = ten(cache["mnist_tr_x"], cache["mnist_tr_y"])
    Xte, yte = ten(cache["mnist_te_x"], cache["mnist_te_y"])
    # D = REAL MNIST train images with SHUFFLED labels (same manifold, opposed labels)
    rng = np.random.default_rng(7_777 + seed)
    yD = rng.permutation(cache["mnist_tr_y"].astype(np.int64))
    XD, yD = ten(cache["mnist_tr_x"], yD)
    return XA, yA, XD, yD, Xte, yte

class SmallCNN(nn.Module):
    def __init__(self, k=32, nclass=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, k, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 14
            nn.Conv2d(k, 2 * k, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7
        )
        self.fc = nn.Linear(2 * k * 7 * 7, nclass)
    def forward(self, x): return self.fc(self.net(x).flatten(1))

def eta_of(t, T_split, T, eta_hi, eta_lo):
    if t < T_split: return eta_hi
    ph = (t - T_split) / max(T - T_split, 1)
    return eta_lo + (eta_hi - eta_lo) * 0.5 * (1.0 + np.cos(np.pi * ph))

def train(XA, yA, XD, yD, win, T=240, T_split=120, eta_hi=0.05, eta_lo=0.005,
          bs=256, seed=0):
    g = torch.Generator(device=DEV).manual_seed(10_000 + seed)
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=eta_hi, momentum=0.0, weight_decay=0.0)
    lossf = nn.CrossEntropyLoss()
    XD_s, yD_s = XD, yD
    nA, nD = len(yA), len(yD_s)
    for t in range(T):
        eta = eta_of(t, T_split, T, eta_hi, eta_lo)
        for pg in opt.param_groups: pg["lr"] = eta
        useD = win is not None and win[0] <= t < win[1]
        if useD:
            X = torch.cat([XA, XD_s]); y = torch.cat([yA, yD_s]); n = nA + nD
        else:
            X, y, n = XA, yA, nA
        perm = torch.randperm(n, generator=g, device=DEV)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(model(X[idx]), y[idx]); loss.backward(); opt.step()
    return model

@torch.no_grad()
def acc_mnist(model, X, y, bs=2000):
    model.eval(); correct = 0
    for i in range(0, len(y), bs):
        logits = model(X[i:i + bs])
        correct += (logits.argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(y)

def main():
    quick = "--quick" in sys.argv
    T, T_split = (8, 4) if quick else (240, 120)
    seeds = [0] if quick else list(range(6))
    t0 = time.time()
    arms = {"never": None, "always": (0, T),
            f"drop@{T_split}": (0, T_split),
            f"inject@{T_split}": (T_split, T),
            "inject@late": (int(0.875 * T), T)}
    rows = {k: [] for k in arms}
    for s in seeds:
        XA, yA, XD, yD, Xte, yte = load_all(DEV, seed=s)  # per-seed shuffled-label D
        if s == 0:
            print(f"[data] A mnist={len(yA)}  D shuffled-label mnist={len(yD)}  test={len(yte)} dev={DEV}", flush=True)
        for name, win in arms.items():
            m = train(XA, yA, XD, yD, win, T=T, T_split=T_split, seed=s)
            a = acc_mnist(m, Xte, yte)
            rows[name].append(a)
            print(f"  seed{s} {name}: acc={a:.4f} ({time.time()-t0:.0f}s)", flush=True)
    med = lambda v: float(np.median(v))
    acc_never, acc_always = med(rows["never"]), med(rows["always"])
    damage = acc_never - acc_always
    rec_split = (med(rows[f"drop@{T_split}"]) - acc_always) / max(damage, 1e-9)
    frac_split = (med(rows[f"inject@{T_split}"]) - acc_always) / max(damage, 1e-9)
    frac_late = (med(rows["inject@late"]) - acc_always) / max(damage, 1e-9)
    out = dict(
        quick=quick, T=T, seeds=seeds, dev=DEV, nA=len(yA), nD=len(yD),
        conflict_block="real MNIST images with shuffled labels (same manifold, opposed labels)",
        acc={k: round(med(v), 4) for k, v in rows.items()},
        acc_all={k: [round(x, 4) for x in v] for k, v in rows.items()},
        damage=round(damage, 4), recovery_drop_at_split=round(rec_split, 3),
        frac_inject_at_split=round(frac_split, 3), frac_inject_late=round(frac_late, 3),
        R1=bool(damage >= 0.01), R2=bool(rec_split >= 0.7), R3=bool(frac_late >= 0.9),
        runtime_min=round((time.time() - t0) / 60, 1),
    )
    tag = "quick" if quick else "full"
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, f"pilot17_realconflict_{tag}_out.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
