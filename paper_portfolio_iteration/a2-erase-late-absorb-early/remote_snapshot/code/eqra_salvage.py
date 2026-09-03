"""
EQRA salvage (Early-Quarantine + Re-Admission) for the CNN + REAL human-noise regime.

Trigger: MGR ac2720d7313c ("only ONE small method salvage"). pilot19 showed real CIFAR-10N
annotator-disagreement damage 0.0427 with drop@120 recovery only 0.240 (hysteresis); diag20
fixed-drop grid shows NO drop time s reaches recovery>=0.7 (free erasure does NOT hold for
CNN+real noise). So salvage does NOT rely on free erasure; it acts DURING the high-LR window.

Method (SALVAGE_PREREG.md, thresholds fixed before seeing results):
  During high-LR epochs (t < split) compute per-epoch cos(g_D, g_A) on a probe (full-batch
  gradient of D block vs A block). If cos < tau_q (=0), quarantine D (weight w_q) for the next
  epoch; re-admit at/after LR decay (t >= split) if the measured cos recovered > tau_r (=0).
  Arms: EQRA-hard (w_q=0) / EQRA-soft (w_q=0.1) / baselines never/always/fixed-drop@120.
  EQRA does NOT assume D is known in practice — here D is the known split to validate the
  mechanism; per-sample gradient-cos scoring is the deployable extension (future work).

Preregistered endpoints (SALVAGE_PREREG.md, DO NOT change after seeing results):
  P1 (positive vs strongest deployable baseline): EQRA acc >= fixed-drop@120 acc + 0.02
  P2 (near oracle): EQRA acc >= never acc - 0.01
  P3 (no collateral on harmless block, tested on controlled pilot18-style arm): acc >= always - 0.005
  Stop line: EQRA acc < fixed-drop@120 + 0.005 -> NEGATIVE-ASSET the method.
Fairness: same 240-epoch budget, same SmallCNN, same WSD, 6 seeds, paired diffs, mean+-std.
"""
import numpy as np, json, time, os, sys
import torch, torch.nn as nn
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p19", os.path.join(HERE, "pilot19_cifar10n.py"))
p19 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p19)
DEV = p19.DEV


def grad_of(model, lossf, X, y):
    """Full-batch gradient vector of the block loss."""
    model.train()
    model.zero_grad()
    loss = lossf(model(X), y)
    loss.backward()
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()])


def train_eqra(XA, yA, XD, yD, arm, T=240, T_split=120, eta_hi=0.05, eta_lo=0.005,
               bs=256, seed=0, tau_q=0.0, tau_r=0.0, w_soft=0.1):
    g = torch.Generator(device=DEV).manual_seed(30_000 + seed)
    torch.manual_seed(seed); np.random.seed(seed)
    model = p19.SmallCNN().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=eta_hi, momentum=0.0, weight_decay=0.0)
    lossf = nn.CrossEntropyLoss()
    nA, nD = len(yA), len(yD)
    cos_log, quar_log = [], []
    quarantined = False
    for t in range(T):
        eta = p19.eta_of(t, T_split, T, eta_hi, eta_lo)
        for pg in opt.param_groups: pg["lr"] = eta
        if arm.startswith("eqra"):
            # measure conflict on current model once per epoch
            if t < T_split:
                gA = grad_of(model, lossf, XA, yA)
                gD = grad_of(model, lossf, XD, yD)
                cos = float(torch.dot(gA, gD) / (gA.norm() * gD.norm() + 1e-12))
                quarantined = cos < tau_q
            else:  # low-LR tail: re-admission check
                gA = grad_of(model, lossf, XA, yA)
                gD = grad_of(model, lossf, XD, yD)
                cos = float(torch.dot(gA, gD) / (gA.norm() * gD.norm() + 1e-12))
                if cos > tau_r:
                    quarantined = False
            cos_log.append(round(float(cos), 4)); quar_log.append(bool(quarantined))
            wD = 0.0 if (arm == "eqra-hard" and quarantined) else (
                w_soft if (arm == "eqra-soft" and quarantined) else 1.0)
        else:
            wD = 1.0
        # build epoch data
        if arm == "never":
            X, y, n = XA, yA, nA
        elif arm == "always":
            X = torch.cat([XA, XD]); y = torch.cat([yA, yD]); n = nA + nD
        elif arm == "drop@120":
            if t < T_split:
                X = torch.cat([XA, XD]); y = torch.cat([yA, yD]); n = nA + nD
            else:
                X, y, n = XA, yA, nA
        else:  # eqra arms: always include D but scaled by wD via loss weighting
            X = torch.cat([XA, XD]); y = torch.cat([yA, yD]); n = nA + nD
        perm = torch.randperm(n, generator=g, device=DEV)
        model.train()
        isD = torch.cat([torch.zeros(nA, dtype=torch.bool, device=DEV),
                         torch.ones(nD, dtype=torch.bool, device=DEV)]) if n == nA + nD else None
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = model(X[idx])
            if isD is not None and wD < 1.0:
                per = nn.functional.cross_entropy(logits, y[idx], reduction="none")
                w = torch.where(isD[idx], torch.full_like(per, wD), torch.ones_like(per))
                loss = (per * w).sum() / w.sum()
            else:
                loss = lossf(logits, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model, cos_log, quar_log


def main():
    quick = "--quick" in sys.argv
    T, T_split = (12, 6) if quick else (240, 120)
    seeds = [0, 1] if quick else list(range(6))
    dfrac = 0.10; nA = 15000
    t0 = time.time()
    arms = ["never", "always", "drop@120", "eqra-hard", "eqra-soft"]
    rows = {k: [] for k in arms}
    cos_logs, quar_frac = {}, {}
    for s in seeds:
        XA, yA, XD, yD, Xte, yte, disagree_frac = p19.load_all(DEV, seed=s, dfrac=dfrac, nA=nA)
        for name in arms:
            m, cos_log, quar_log = train_eqra(XA, yA, XD, yD, name, T=T, T_split=T_split, seed=s)
            a = p19.acc_eval(m, Xte, yte)
            rows[name].append(a)
            if name.startswith("eqra"):
                cos_logs[f"{name}_s{s}"] = cos_log
                quar_frac[f"{name}_s{s}"] = round(sum(quar_log) / max(len(quar_log), 1), 3)
            print(f"  seed{s} {name}: acc={a:.4f} ({time.time()-t0:.0f}s)", flush=True)
    med = lambda v: float(np.median(v)); mean = lambda v: float(np.mean(v)); std = lambda v: float(np.std(v))
    acc = {k: med(rows[k]) for k in rows}
    # preregistered endpoints
    P1 = {k: acc[k] >= acc["drop@120"] + 0.02 for k in ("eqra-hard", "eqra-soft")}
    P2 = {k: acc[k] >= acc["never"] - 0.01 for k in ("eqra-hard", "eqra-soft")}
    out = dict(
        quick=quick, T=T, seeds=seeds, dev=DEV, nA=nA, nD=len(yD),
        method="EQRA early-quarantine+re-admission (cos(g_D,g_A)<tau_q=0 quarantine, re-admit if cos>tau_r=0)",
        conflict_block="CIFAR-10N REAL human annotator disagreement (aggre_label), NOT controlled corruption",
        acc_median={k: round(acc[k], 4) for k in rows},
        acc_mean_std={k: [round(mean(v), 4), round(std(v), 4)] for k, v in rows.items()},
        acc_all={k: [round(x, 4) for x in v] for k, v in rows.items()},
        paired_diff_vs_drop120={k: round(acc[k] - acc["drop@120"], 4) for k in ("eqra-hard", "eqra-soft")},
        paired_diff_vs_never={k: round(acc[k] - acc["never"], 4) for k in ("eqra-hard", "eqra-soft")},
        quarantine_frac=quar_frac,
        P1_pass={k: bool(v) for k, v in P1.items()},
        P2_pass={k: bool(v) for k, v in P2.items()},
        cos_logs=cos_logs,
        runtime_min=round((time.time() - t0) / 60, 1),
    )
    tag = "quick" if quick else "full"
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, f"eqra_salvage_{tag}_out.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
