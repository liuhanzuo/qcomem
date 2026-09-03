"""
Phase-C pilot18 (REALISTIC-SCALE, CONTROLLED corruption): does EraseLateAbsorbEarly
give a practical detox-window benefit when the corrupt block D is SMALL (~3% of A),
the realistic data-curation regime (only a few mislabeled examples), vs pilot17's
full-size shuffled-label D (100% of A = adversarial stress test, not realistic).

  LABELING (r783, per human correction 67755759b223): this arm is
  **real-MNIST pixels + CONTROLLED cyclic-label corruption**
  (y->(y+1)%10 on a random 3% subset). The 3% corruption is a CONTROLLED
  experimental condition, NOT real human annotator noise. Real per-annotator
  label-noise datasets (CIFAR-10N, ChaosNLI) were unreachable from this machine
  (DNS/404/401, see RESEARCH_LOG r780); any claim about real annotator
  uncertainty is out of scope for this arm and deferred to limitations.

  A (clean)  = MNIST train (0-9), eval on MNIST test.
  D (conflict)= SMALL subset of REAL MNIST train images with CONTROLLED
                cyclic-flipped labels (per-class cyclic flip y->(y+1)%10 on a
                random ~3% subset). Same input manifold, opposed labels,
                near-dup gradient conflict, at the realistic scale where detox
                actually matters.

Model: same SmallCNN as pilot17. Plain mini-batch SGD, cosine WSD tail.

Arms (mirror pilot17): never / always / drop@split / inject@split / inject@late.

Preregistered:
  R1 object exists at realistic scale: damage >= 0.005 (0.5% MNIST acc, small block)
  R2 erasure-free survives: recovery(drop@split) >= 0.7
  R3 T3 recency/budget: frac_late >= 0.9
  R4 report-only: frac_split partial gradient
Method hook (Gate C): if R1&R2, the "detection window extends to just before LR
  decay at no recovery cost" benefit holds at the realistic few-mislabeled scale.
"""
import numpy as np, json, time, os, sys
import torch, torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
DROOT = os.path.expanduser("~/.cache/mnist_fashion")

URLS = {
    "mnist_tr_x": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "mnist_tr_y": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "mnist_te_x": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "mnist_te_y": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

def _fetch(url):
    import urllib.request, gzip
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return gzip.decompress(urllib.request.urlopen(req, timeout=60).read())

def _idx_images(b):
    n = int.from_bytes(b[4:8], "big"); r = int.from_bytes(b[8:12], "big"); c = int.from_bytes(b[12:16], "big")
    return np.frombuffer(b[16:], np.uint8).reshape(n, r, c)

def _idx_labels(b):
    n = int.from_bytes(b[4:8], "big"); return np.frombuffer(b[8:8 + n], np.uint8)

def load_all(device, seed=0, dfrac=0.03):
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
        x = torch.tensor(img.astype(np.float32) / 255.0).unsqueeze(1)
        x = (x - 0.1307) / 0.3081
        return x.to(device), torch.tensor(lab.astype(np.int64), device=device)
    XA, yA = ten(cache["mnist_tr_x"], cache["mnist_tr_y"])
    Xte, yte = ten(cache["mnist_te_x"], cache["mnist_te_y"])
    # D = SMALL subset of REAL MNIST train images with FLIPPED labels (cyclic y->(y+1)%10)
    rng = np.random.default_rng(31_000 + seed)
    nA = len(cache["mnist_tr_y"]); nD = int(dfrac * nA)
    sel = rng.choice(nA, size=nD, replace=False)
    XD_img = cache["mnist_tr_x"][sel]
    yD_lab = (cache["mnist_tr_y"][sel].astype(np.int64) + 1) % 10  # cyclic flip
    XD, yD = ten(XD_img, yD_lab)
    return XA, yA, XD, yD, Xte, yte

class SmallCNN(nn.Module):
    def __init__(self, k=32, nclass=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, k, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(k, 2 * k, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(2 * k * 7 * 7, nclass)
    def forward(self, x): return self.fc(self.net(x).flatten(1))

def eta_of(t, T_split, T, eta_hi, eta_lo):
    if t < T_split: return eta_hi
    ph = (t - T_split) / max(T - T_split, 1)
    return eta_lo + (eta_hi - eta_lo) * 0.5 * (1.0 + np.cos(np.pi * ph))

def train(XA, yA, XD, yD, win, T=240, T_split=120, eta_hi=0.05, eta_lo=0.005, bs=256, seed=0):
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
            opt.zero_grad(); loss = lossf(model(X[idx]), y[idx]); loss.backward(); opt.step()
    return model

@torch.no_grad()
def acc_mnist(model, X, y, bs=2000):
    model.eval(); correct = 0
    for i in range(0, len(y), bs):
        correct += (model(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(y)

def main():
    quick = "--quick" in sys.argv
    T, T_split = (12, 6) if quick else (240, 120)
    seeds = [0, 1] if quick else list(range(6))
    dfrac = 0.03
    t0 = time.time()
    arms = {"never": None, "always": (0, T),
            f"drop@{T_split}": (0, T_split),
            f"inject@{T_split}": (T_split, T),
            "inject@late": (int(0.875 * T), T)}
    rows = {k: [] for k in arms}
    for s in seeds:
        XA, yA, XD, yD, Xte, yte = load_all(DEV, seed=s, dfrac=dfrac)
        if s == seeds[0]:
            print(f"[data] A mnist={len(yA)}  D flipped-label subset={len(yD)} ({dfrac:.0%})  test={len(yte)} dev={DEV}", flush=True)
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
        quick=quick, T=T, seeds=seeds, dev=DEV, nA=len(yA), nD=len(yD), dfrac=dfrac,
        conflict_block="real-MNIST pixels + CONTROLLED cyclic-label corruption y->(y+1)%10 on a random 3% subset (controlled condition, NOT real annotator noise)",
        acc={k: round(med(v), 4) for k, v in rows.items()},
        acc_all={k: [round(x, 4) for x in v] for k, v in rows.items()},
        damage=round(damage, 4), recovery_drop_at_split=round(rec_split, 3),
        frac_inject_at_split=round(frac_split, 3), frac_inject_late=round(frac_late, 3),
        R1=bool(damage >= 0.005), R2=bool(rec_split >= 0.7), R3=bool(frac_late >= 0.9),
        runtime_min=round((time.time() - t0) / 60, 1),
    )
    tag = "quick" if quick else "full"
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, f"pilot18_smallconflict_{tag}_out.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
