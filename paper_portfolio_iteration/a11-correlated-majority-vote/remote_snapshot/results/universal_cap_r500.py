#!/usr/bin/env python3
"""r500: universal repair budget law — pure re-analysis of r498e full sweep.

Reads earlystop_drift_r498/full_sweep_r498e_result.json (72 carrier x m x alpha
cells x 12 caps, exact LP radii; zero new data, zero GPU, zero TEST readout)
and extracts:
  S1 per-cell critical cap (smallest scanned cap with tau*=1): 72/72 = 0.01.
  S2 universal sufficiency: cap=0.01 => tau*=1 in 72/72 cells; cap=0.015 fails
     exactly the 18 alpha=0.01 cells (6 per carrier), all other 54 pass.
  S3 cost of the universal cap: votes_ratio at cap=0.01 per carrier
     (min/max), max base_S at cap=0.01.
  S4 regen anchor: tau_orig column matches r491/r498 frozen grid (report count).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "earlystop_drift_r498", "full_sweep_r498e_result.json")
d = json.load(open(SRC))
caps_asc = sorted(d["caps"])

out = {"source": SRC, "seed": d["seed"], "definition": "critical cap = smallest scanned cap with tau*>=1-1e-12",
       "S1_critical_caps": {}, "S2_cap0015_failures": [], "S3_cost_at_cap001": {}, "S4_tau_orig_regen_mismatch": None}

n1 = 0
for cname, cd in d["carriers"].items():
    for cell in cd["cells"]:
        crit = None
        for cp in caps_asc:
            if cell["per_cap"][str(cp)]["tau"] >= 1.0 - 1e-12:
                crit = cp
                break
        out["S1_critical_caps"].setdefault(str(crit), 0)
        out["S1_critical_caps"][str(crit)] += 1
        n1 += 1
        t15 = cell["per_cap"]["0.015"]["tau"]
        if t15 < 1.0 - 1e-12:
            out["S2_cap0015_failures"].append({"carrier": cname, "m": cell["m"], "alpha": cell["alpha"], "tau": t15})

for cname, cd in d["carriers"].items():
    vr = [cell["per_cap"]["0.01"]["votes_ratio"] for cell in cd["cells"]]
    bs = [cell["per_cap"]["0.01"]["base_S"] for cell in cd["cells"]]
    out["S3_cost_at_cap001"][cname] = {"votes_ratio_min": min(vr), "votes_ratio_max": max(vr), "base_S_max": max(bs)}

# S4: tau_orig at (m_full) cells must equal r481/482 radii (frozen grid anchor): just count cells present
out["S4_tau_orig_regen_mismatch"] = d.get("checks", {}).get("regen_mismatch", None)
out["checks"] = {
    "S1_all_critical_001": out["S1_critical_caps"] == {"0.01": n1} and n1 == 72,
    "S2_failures_all_alpha001": len(out["S2_cap0015_failures"]) == 18 and all(f["alpha"] == 0.01 for f in out["S2_cap0015_failures"]),
    "S2_failures_6_per_carrier": all(sum(1 for f in out["S2_cap0015_failures"] if f["carrier"] == c) == 6 for c in d["carriers"]),
}
out["checks"]["ALL_PASS"] = all(out["checks"].values())

dst = os.path.join(HERE, "universal_cap_r500_result.json")
json.dump(out, open(dst, "w"), indent=1)
print(json.dumps(out["checks"], indent=1))
print("S1:", out["S1_critical_caps"])
print("S3:", json.dumps(out["S3_cost_at_cap001"], indent=1))
