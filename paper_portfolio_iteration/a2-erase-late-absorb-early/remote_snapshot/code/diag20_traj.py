"""
diag20: trajectory diagnosis of pilot19 hysteresis (MGR ac2720d7313c, DIAG20_PREREG.md).

Same data/split as pilot19 (CIFAR-10N real human disagreement, seeds 0/1/2).
Instrumented arms (per-epoch test_acc, d_loss, d_acc, a_loss; every 10 ep
cos(g_D,g_A) and representation drift vs final never features):
  never / always / drop@120 / inject@late
Fixed-drop grid (final acc only): s in {90,135,150,165,180,210}.
"""
import numpy as np, json, time, os, sys
import torch, torch.nn as nn
from pilot19_cifar10n import load_all, SmallCNN, eta_of, acc_eval, DEV, HERE

lossf = nn.CrossEntropyLoss()

@torch.no_grad()
def subset_metrics(model, X, y, bs=4000):
    model.eval(); tot, corr, loss = 0, 0, 0.0
    for i in range(0, len(y), bs):
        logits = model(X[i:i+bs]); l = lossf(logits, y[i:i+bs])
        loss += l.item() * len(y[i:i+bs]); corr += (logits.argmax(1) == y[i:i+bs]).sum().item()
        tot += len(y[i:i+bs])
    return loss / tot, corr / tot

def fullbatch_grad(model, X, y):
    model.eval(); model.zero_grad(set_to_none=True)
    lossf(model(X), y).backward()
    return torch.cat([p.grad.detach().flatten().clone() for p in model.parameters()])

@torch.no_grad()
def feats(model, X):
    model.eval(); return model.net(X).flatten(1)

def train_final(XA, yA, XD, yD, win, Xte, yte, T=240, T_split=120, eta_hi=0.05,
                eta_lo=0.005, bs=256, seed=0):
    g = torch.Generator(device=DEV).manual_seed(20_000 + seed)
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=eta_hi, momentum=0.0, weight_decay=0.0)
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
    return acc_eval(model, Xte, yte)

def main():
    quick = "--quick" in sys.argv
    T, T_split = (12, 6) if quick else (240, 120)
    seeds = [0] if quick else [0, 1, 2]
    drop_grid = [6] if quick else [90, 135, 150, 165, 180, 210]
    dfrac, nA = 0.10, 15000
    t0 = time.time()
    seed_out = []
    for s in seeds:
        XA, yA, XD, yD, Xte, yte, _ = load_all(DEV, seed=s, dfrac=dfrac, nA=nA)
        probe = Xte[:1000]
        # run never first and keep its final model features as drift reference
        tr_never, never_model = train_traj_ret(XA, yA, XD, yD, None, Xte, yte, probe, None, T=T, T_split=T_split, seed=s)
        ref_feat = feats(never_model, probe).detach()
        tr = {"never": tr_never}
        for name, win in [("always", (0, T)), (f"drop@{T_split}", (0, T_split)),
                          ("inject@late", (int(0.875 * T), T))]:
            tr[name], _ = train_traj_ret(XA, yA, XD, yD, win, Xte, yte, probe, ref_feat, T=T, T_split=T_split, seed=s)
            print(f"  seed{s} {name} done ({time.time()-t0:.0f}s)", flush=True)
        drops = {}
        for ds in drop_grid:
            drops[ds] = train_final(XA, yA, XD, yD, (0, ds), Xte, yte, T=T, T_split=T_split, seed=s)
            print(f"  seed{s} drop@{ds} acc={drops[ds]:.4f} ({time.time()-t0:.0f}s)", flush=True)
        seed_out.append({"seed": s, "traj": tr, "drops": drops})
    # aggregate medians across seeds
    def med_curve(key, metric):
        cur = [so["traj"][key][metric] for so in seed_out]
        return [float(np.median([c[i] for c in cur])) for i in range(T)]
    arms = ["never", "always", f"drop@{T_split}", "inject@late"]
    agg = {a: {m: med_curve(a, m) for m in ["test_acc", "d_loss", "d_acc", "a_loss"]} for a in arms}
    cos_agg, drift_agg = {}, {}
    for a in arms:
        keys = sorted(seed_out[0]["traj"][a]["cos"].keys(), key=int)
        cos_agg[a] = {k: float(np.median([so["traj"][a]["cos"][k] for so in seed_out])) for k in keys}
        drift_agg[a] = {k: float(np.median([so["traj"][a]["drift"][k] for so in seed_out])) for k in keys}
    drop_med = {str(ds): float(np.median([so["drops"][ds] for so in seed_out])) for ds in drop_grid}
    acc_never = agg["never"]["test_acc"][-1]; acc_always = agg["always"]["test_acc"][-1]
    dmg = acc_never - acc_always
    drop_rec = {ds: (a - acc_always) / max(dmg, 1e-9) for ds, a in drop_med.items()}
    out = dict(quick=quick, T=T, seeds=seeds, dmg=round(dmg, 4),
               acc_never=round(acc_never, 4), acc_always=round(acc_always, 4),
               traj=agg, cos=cos_agg, drift=drift_agg,
               drop_grid_acc={k: round(v, 4) for k, v in drop_med.items()},
               drop_recovery={k: round(v, 3) for k, v in drop_rec.items()},
               runtime_min=round((time.time() - t0) / 60, 1))
    tag = "quick" if quick else "full"
    json.dump(out, open(os.path.join(HERE, f"diag20_traj_{tag}_out.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "traj"}, indent=1)[:2000])

def train_traj_ret(XA, yA, XD, yD, win, Xte, yte, probe, ref_feat, T, T_split,
                   eta_hi=0.05, eta_lo=0.005, bs=256, seed=0):
    g = torch.Generator(device=DEV).manual_seed(20_000 + seed)
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=eta_hi, momentum=0.0, weight_decay=0.0)
    nA, nD = len(yA), len(yD)
    traj = {"test_acc": [], "d_loss": [], "d_acc": [], "a_loss": [], "cos": {}, "drift": {}}
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
        traj["test_acc"].append(acc_eval(model, Xte, yte))
        dl, da = subset_metrics(model, XD, yD); al, _ = subset_metrics(model, XA, yA)
        traj["d_loss"].append(dl); traj["d_acc"].append(da); traj["a_loss"].append(al)
        if (t % 10 == 9 or t == T - 1) and ref_feat is not None:
            gD = fullbatch_grad(model, XD, yD); gA = fullbatch_grad(model, XA, yA)
            traj["cos"][str(t)] = float(torch.dot(gD, gA) / (gD.norm() * gA.norm() + 1e-12))
            f = feats(model, probe)
            traj["drift"][str(t)] = float((f - ref_feat).norm(dim=1).mean())
    return traj, model

if __name__ == "__main__":
    main()
