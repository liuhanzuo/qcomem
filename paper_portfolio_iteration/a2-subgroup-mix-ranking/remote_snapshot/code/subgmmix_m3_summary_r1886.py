"""Canonical M3 summary (r1886): committed rate / cert validity / defer cost, per
carrier x frac x policy x tau, averaged over 3 seeds.

Defer = abstained rows. We report TWO fallbacks on abstained rows:
  - hardpick : argmin_i pt_i(w) (per-w point-estimate, the M2 baseline)  [negative control]
  - anchor   : a SINGLE robust-mixture fallback model chosen PRE-HOC from the FIT split
               (argmin over FIT-best per-model worst-case mixture risk over the w-grid),
               deployed on every abstained w.  Its true OUTER regret is the defer cost.
Honest distinction: hardpick is per-w (can still overturn); anchor is fixed and never
peeks CAL/OUTER (chosen from FIT only), so its defer cost is a tight, honest lower-risk
default that does NOT claim the benefit of a per-w query.

Also flags global cert-validity: committed rows must satisfy true_regret<=tau.
PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX  ROUND r1886  (summary; pure CPU/repo jsonio)
"""
import json, os, glob, numpy as np

OUT = "subgroup_mix_ranking/results/SUBGMIX_M3_SUMMARY_R1886.json"
TAU = 0.04
CACHE = "subgroup_mix_ranking/results/m3_cache"
RCD = "subgroup_mix_ranking/results/SUBGMIX_M3_BUDGET_R1886.json"
TOOL = os.path.join(os.path.dirname(__file__), "..", "..",
                    "subgroup_mix_ranking", "code")


def true_outer_per_w(grid, outer, G, Mnames):
    return None  # helper unused


def main():
    agg = {}
    for f in sorted(glob.glob(os.path.join(CACHE, "recs_*.npz"))):
        base = os.path.basename(f)              # recs_digits_s0.npz
        stem = base.replace("recs_", "")         # digits_s0.npz
        name = stem.rsplit("_s", 1)[0]           # digits
        seed = stem.rsplit("_s", 1)[1].split(".")[0]  # 0
        z = np.load(f, allow_pickle=True)
        recs = z["recs"].item()
        grid = z["grid"].item()
        zo = z["outer"].item()
        outer = {m: {int(k): float(v) for k, v in d.item().items()}
                 for m, d in zo.items()}
        Mnames = list(outer.keys())
        G = len(outer[Mnames[0]])

        # FIT-precomputed robust anchor (uses only the already-labeled FIT split)
        zf = np.load(os.path.join(CACHE, f"subgmmix_m3_{name}_s{seed}.npz"),
                     allow_pickle=True)
        yf = zf["yf"]
        u = {g: (yf == g).sum() / len(yf) for g in range(G)}
        fitw = {}
        for m in zf["Mnames"]:
            fe = zf[f"fe_{m}"].astype(float)
            pte = {g: fe[yf == g].mean() if (yf == g).sum() else 0.0
                   for g in range(G)}
            fitw[m] = max(sum(wi["w"][g] * pte[g] for g in range(G))
                          for wi in grid)
        anchor = min(fitw, key=fitw.get)

        for (frac, pol), rows in recs.items():
            for tau in [0.04]:
                comm = [r for r in rows if r["UB"] <= tau]
                abst = [r for r in rows if r["UB"] > tau]
                cr = len(comm) / len(rows)
                cv = float(np.mean([r["true_regret"] <= tau + 1e-9
                                    for r in comm])) if comm else None
                # defer costs
                hard_m = hard_x = anch_m = anch_x = None
                if abst:
                    hregs = [r["true_regret"] for r in abst]
                    hard_m = float(np.mean(hregs)); hard_x = float(np.max(hregs))
                    aregs = []
                    for r in abst:
                        wi = next(w2 for w2 in grid if w2["name"] == r["w"])
                        w = wi["w"]
                        tr = {m: sum(w[g] * outer[m][g] for g in range(G))
                              for m in Mnames}
                        aregs.append(tr[anchor] - min(tr.values()))
                    anch_m = float(np.mean(aregs)); anch_x = float(np.max(aregs))
                key = (name, frac, pol)
                agg.setdefault(key, []).append(
                    dict(carrier=name, frac=frac, policy=pol, seed=seed,
                         committed_rate=cr, cert_validity=cv,
                         n_w=len(rows), n_abst=len(abst),
                         hardpick_defer_mean=hard_m, hardpick_defer_max=hard_x,
                         anchor_defer_mean=anch_m, anchor_defer_max=anch_x,
                         anchor=anchor))
    # aggregate
    out = {}
    for (name, frac, pol), vals in sorted(
            agg.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        crs = [v["committed_rate"] for v in vals]
        cvs = [v["cert_validity"] for v in vals if v["cert_validity"] is not None]
        hard_m = [v["hardpick_defer_mean"] for v in vals if v["hardpick_defer_mean"] is not None]
        hard_x = [v["hardpick_defer_max"] for v in vals if v["hardpick_defer_max"] is not None]
        anch_m = [v["anchor_defer_mean"] for v in vals if v["anchor_defer_mean"] is not None]
        anch_x = [v["anchor_defer_max"] for v in vals if v["anchor_defer_max"] is not None]
        out[f"({frac},{pol})_{name}"] = {
            "carrier": name, "frac": frac, "policy": pol,
            "committed_rate": round(float(np.mean(crs)), 4),
            "cert_validity": round(float(np.mean(cvs)), 4) if cvs else None,
            "hardpick_defer_mean": round(float(np.mean(hard_m)), 4) if hard_m else None,
            "hardpick_defer_max": round(float(np.max(hard_x)), 4) if hard_x else None,
            "anchor_defer_mean": round(float(np.mean(anch_m)), 4) if anch_m else None,
            "anchor_defer_max": round(float(np.max(anch_x)), 4) if anch_x else None,
            "anchor": vals[0]["anchor"], "n_seeds": len(vals)}
    meta = {"tau": TAU, "note": ("tau=0.04. hardpick/per-w negative control; "
                                 "anchor=FIT-pre-chosen robust fallback, no CAL/OUTER peek.")}
    with open(OUT, "w") as fh:
        json.dump({"project": "A2_SAFE_MODEL_RANKING_SUBGROUP_MIX",
                   "round": "r1886", "agg": out, "meta": meta}, fh, indent=2)
    print("saved", OUT, "keys", len(out))


if __name__ == "__main__":
    main()