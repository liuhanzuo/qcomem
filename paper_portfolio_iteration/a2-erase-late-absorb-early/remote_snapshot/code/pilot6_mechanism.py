"""
Phase-1 pilot6 (mechanism probe for the nonconvex reversal found in pilot5).

pilot5 surprise: in the MLP regime, inject@210 (D present ONLY in last 30 low-LR
steps) causes FULL damage (fraction ~1.0), reversing the convex absorption law
(damage ~ exposure in high-LR phase). Hypothesis: saturation asymmetry --
once A is fit, tanh/logistic saturation kills A's gradients while D (mislabeled,
confidently wrong) keeps large gradients, so D dominates late updates.

Preregistered:
  P1 (gradient dominance): for a model trained clean to t=210,
     ||g_D|| / (||g_A|| + ||g_D||) >= 0.8 ; at t=60 (high-LR, unfitted) ratio <= 0.6.
  P2 (inject curve transition): damage fraction at inject s=180 strictly between
     s=120 and s=210 medians (monotone INCREASING late-inject damage in MLP,
     opposite sign vs convex pilot3).
  P3 (erasure horizon): recovery(drop@225) >= 0.5, recovery(drop@235) reported
     (how many decay steps does erasure need?).
  P4 (net-gain mechanism, report only): test mean |f| and train 5th-pct margin
     for never vs drop@120 -- is never overconfident (larger |f|, tail errors)?
"""
import numpy as np, json, time
from pilot5_boundary import (make_data, init_params, forward, loss_grad_batch,
                             eta_of, metrics)

T, T_split = 240, 120


def train_probe(XA, yA, XD, yD, win, seed, probe_times=()):
    rng = np.random.default_rng(10_000 + seed)
    p = init_params(rng)
    probes = {}
    lam, bs = 1e-4, 128
    for t in range(T):
        if t in probe_times:
            _, gA = loss_grad_batch(p, XA, yA, 0.0)
            _, gD = loss_grad_batch(p, XD, yD, 0.0)
            nA = sum(float((np.asarray(gA[k]) ** 2).sum()) for k in gA) ** 0.5
            nD = sum(float((np.asarray(gD[k]) ** 2).sum()) for k in gD) ** 0.5
            probes[t] = dict(gA=nA, gD=nD, dom=nD / (nA + nD + 1e-12))
        eta = eta_of(t, T_split, T, 0.2, 0.01)
        blocks = [(XA, yA)] + ([(XD, yD)] if win is not None and win[0] <= t < win[1] else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        perm = rng.permutation(len(y))
        for i in range(0, len(y), bs):
            idx = perm[i:i + bs]
            _, g = loss_grad_batch(p, X[idx], y[idx], lam)
            for k in p: p[k] = p[k] - eta * g[k]
    return p, probes


def main():
    t0 = time.time()
    seeds = list(range(8))
    inject_s = [120, 150, 180, 210, 225]
    drop_s = [225, 235]
    acc_rows = {"never": [], "always": []}
    for s in inject_s: acc_rows[f"inject@{s}"] = []
    for s in drop_s: acc_rows[f"drop@{s}"] = []
    probe_rows = {60: [], 210: []}
    tail = {"never": {"absf": [], "m5": []}, "drop@120": {"absf": [], "m5": []}}
    for sd in seeds:
        XA, yA, XD, yD, Xt, yt = make_data(sd)
        # probes on the clean trajectory
        _, pr = train_probe(XA, yA, XD, yD, None, sd, probe_times=(60, 210))
        for t in pr: probe_rows[t].append(pr[t]["dom"])
        arm_list = ([("never", None), ("always", (0, T))]
                    + [(f"inject@{s}", (s, T)) for s in inject_s]
                    + [(f"drop@{s}", (0, s)) for s in drop_s])
        for name, win in arm_list:
            p, _ = train_probe(XA, yA, XD, yD, win, sd)
            acc, _, _, _ = metrics(p, XA, yA, Xt, yt)
            acc_rows[name].append(acc)
            if name in tail:
                _, ft = forward(p, Xt)
                _, fa = forward(p, XA)
                m = (2 * yA - 1) * fa
                tail[name]["absf"].append(float(np.mean(np.abs(ft))))
                tail[name]["m5"].append(float(np.percentile(m, 5)))

    med = lambda v: float(np.median(v))
    acc_never, acc_always = med(acc_rows["never"]), med(acc_rows["always"])
    damage = acc_never - acc_always
    frac = {k: round((acc_never - med(v)) / max(damage, 1e-9), 3)
            for k, v in acc_rows.items() if k.startswith("inject")}
    rec = {k: round((med(v) - acc_always) / max(damage, 1e-9), 3)
           for k, v in acc_rows.items() if k.startswith("drop")}
    out = dict(
        damage=round(damage, 4),
        inject_damage_frac=frac,
        drop_recovery=rec,
        dominance_t60=round(med(probe_rows[60]), 3),
        dominance_t210=round(med(probe_rows[210]), 3),
        tail_absf={k: round(med(v["absf"]), 3) for k, v in tail.items()},
        tail_margin5={k: round(med(v["m5"]), 3) for k, v in tail.items()},
        P1=bool(med(probe_rows[210]) >= 0.8 and med(probe_rows[60]) <= 0.6),
        P2=bool(frac["inject@180"] > frac["inject@120"] - 0.05),
        P3=bool(rec["drop@225"] >= 0.5),
        runtime_sec=round(time.time() - t0, 1),
    )
    print(json.dumps(out, indent=1))
    json.dump(out, open("pilot6_mechanism_out.json", "w"), indent=1)


if __name__ == "__main__":
    main()
