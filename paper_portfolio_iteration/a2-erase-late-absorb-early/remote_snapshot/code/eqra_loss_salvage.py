"""
EQRA-loss salvage: per-sample loss early-quarantine (fallback for EQRA-cos).

Trigger: EQRA-cos failed (P1/P2 fail) because block-level cos(g_D,g_A) is positive during the
warmup window (epoch<60, diag20 H-b REFUTED) so quarantine triggers too late, after the damage
is already absorbed into the representation attractor. Fallback signal = per-sample loss: real
human-noise examples have high loss during high-LR warmup (model has not yet fit the noise
label). This is the classic noise-detection signal (Arazo/MentorNet/Co-teaching lineage) —
novelty here is the theory-driven closure (budget law predicts which block gets absorbed;
early loss-quarantine prevents absorption), NOT the detection signal itself.

Preregistered (SALVAGE_LOSS_PREREG.md, identical endpoints to SALVAGE_PREREG, DO NOT change):
  P1: EQRA-loss acc >= fixed-drop@120 + 0.02 ; P2: >= never - 0.01 ; P3: no collateral on
  controlled arm. Stop line: < drop@120 + 0.005 -> NEGATIVE-ASSET.
  q_frac=0.10 (= true D fraction, fixed a priori, not tuned on results).
  Framework (quarantine in warmup, re-admit at LR decay if loss re-enters clean range) is the
  same as EQRA-cos; ONLY the detection signal differs.
"""
import numpy as np, json, time, os, sys
import torch, torch.nn as nn
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p19", os.path.join(HERE, "pilot19_cifar10n.py"))
p19 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p19)
DEV = p19.DEV


@torch.no_grad()
def per_sample_loss(model, X, y, bs=2000):
    model.eval()
    out = np.empty(len(y), dtype=np.float32)
    for i in range(0, len(y), bs):
        logits = model(X[i:i + bs])
        out[i:i + bs] = nn.functional.cross_entropy(
            logits, y[i:i + bs], reduction="none").cpu().numpy()
    return out


def train_eqra_loss(XA, yA, XD, yD, arm, T=240, T_split=120, eta_hi=0.05, eta_lo=0.005,
                    bs=256, seed=0, q_frac=0.10):
    g = torch.Generator(device=DEV).manual_seed(40_000 + seed)
    torch.manual_seed(seed); np.random.seed(seed)
    model = p19.SmallCNN().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=eta_hi, momentum=0.0, weight_decay=0.0)
    lossf = nn.CrossEntropyLoss()
    nA, nD = len(yA), len(yD)
    Xall = torch.cat([XA, XD]); yall = torch.cat([yA, yD]); nall = nA + nD
    isD_all = torch.cat([torch.zeros(nA, dtype=torch.bool, device=DEV),
                         torch.ones(nD, dtype=torch.bool, device=DEV)])
    quar_mask = torch.zeros(nall, dtype=torch.bool, device=DEV)
    quarantine_frac_log = []
    for t in range(T):
        eta = p19.eta_of(t, T_split, T, eta_hi, eta_lo)
        for pg in opt.param_groups: pg["lr"] = eta
        if arm == "eqra-loss":
            if t < T_split:
                # mark highest-loss q_frac of all samples for quarantine next epoch
                ls = per_sample_loss(model, Xall, yall)
                k = max(int(q_frac * nall), 1)
                thr = np.partition(ls, -k)[-k]
                quar_mask = torch.tensor(ls >= thr, device=DEV)
            else:
                # re-admission: drop quarantine flag if sample loss back within clean range
                ls = per_sample_loss(model, Xall, yall)
                clean_ref = np.median(ls[:nA])  # median clean loss as reference
                still_bad = torch.tensor(ls > clean_ref, device=DEV)
                quar_mask = quar_mask & still_bad
            quarantine_frac_log.append(round(float(quar_mask.float().mean()), 4))
        # select active set
        if arm == "never":
            X, y = XA, yA
        elif arm == "always":
            X, y = Xall, yall
        elif arm == "drop@120":
            X, y = (Xall, yall) if t < T_split else (XA, yA)
        else:  # eqra-loss: exclude quarantined samples
            keep = ~quar_mask
            X, y = Xall[keep], yall[keep]
        n = len(y)
        perm = torch.randperm(n, generator=g, device=DEV)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(model(X[idx]), y[idx]); loss.backward(); opt.step()
    return model, quarantine_frac_log


def main():
    quick = "--quick" in sys.argv
    T, T_split = (12, 6) if quick else (240, 120)
    seeds = [0, 1] if quick else list(range(6))
    dfrac = 0.10; nA = 15000
    t0 = time.time()
    arms = ["never", "always", "drop@120", "eqra-loss"]
    rows = {k: [] for k in arms}
    qlog = {}
    for s in seeds:
        XA, yA, XD, yD, Xte, yte, _ = p19.load_all(DEV, seed=s, dfrac=dfrac, nA=nA)
        for name in arms:
            m, ql = train_eqra_loss(XA, yA, XD, yD, name, T=T, T_split=T_split, seed=s)
            a = p19.acc_eval(m, Xte, yte)
            rows[name].append(a)
            if name == "eqra-loss":
                qlog[f"s{s}"] = ql
            print(f"  seed{s} {name}: acc={a:.4f} ({time.time()-t0:.0f}s)", flush=True)
    med = lambda v: float(np.median(v)); mean = lambda v: float(np.mean(v)); std = lambda v: float(np.std(v))
    acc = {k: med(rows[k]) for k in rows}
    out = dict(
        quick=quick, T=T, seeds=seeds, dev=DEV, nA=nA, nD=len(yD),
        method="EQRA-loss: per-sample-loss early-quarantine (highest q_frac=0.10 loss quarantined in warmup, re-admit at LR decay)",
        conflict_block="CIFAR-10N REAL human annotator disagreement (aggre_label), NOT controlled corruption",
        acc_median={k: round(acc[k], 4) for k in rows},
        acc_mean_std={k: [round(mean(rows[k]), 4), round(std(rows[k]), 4)] for k in rows},
        acc_all={k: [round(x, 4) for x in rows[k]] for k in rows},
        diff_vs_drop120=round(acc["eqra-loss"] - acc["drop@120"], 4),
        diff_vs_never=round(acc["eqra-loss"] - acc["never"], 4),
        P1_pass=bool(acc["eqra-loss"] >= acc["drop@120"] + 0.02),
        P2_pass=bool(acc["eqra-loss"] >= acc["never"] - 0.01),
        quarantine_frac_log=qlog,
        runtime_min=round((time.time() - t0) / 60, 1),
    )
    tag = "quick" if quick else "full"
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, f"eqra_loss_{tag}_out.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
