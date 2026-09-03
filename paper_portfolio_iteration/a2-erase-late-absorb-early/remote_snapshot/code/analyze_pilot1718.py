"""Analyze pilot17 (real-pixel 100% shuffled-label conflict) and pilot18
(real-MNIST pixels + CONTROLLED 3% cyclic-label corruption) full-run outputs
against their preregistered R1-R3 criteria. Prints verdict + writes
pilot1718_verdict.json. Run after both pilots finish (or on partial logs with
--from-log to parse the running log instead of the JSON)."""
import json, os, sys, re, statistics

HERE = os.path.dirname(os.path.abspath(__file__))

def from_json(path):
    return json.load(open(path))

def from_log(path, arms):
    """Parse 'seedK NAME: acc=X' lines; median per arm; same math as the pilot scripts."""
    rows = {a: [] for a in arms}
    pat = re.compile(r"seed\d+ (.+?): acc=([0-9.]+)")
    for line in open(path):
        m = pat.search(line)
        if m and m.group(1) in rows:
            rows[m.group(1)].append(float(m.group(2)))
    if not all(rows.values()):
        return None
    med = {k: statistics.median(v) for k, v in rows.items()}
    nseeds = max(len(v) for v in rows.values())
    return med, nseeds, rows

def verdict(name, med, floors):
    acc_never, acc_always = med["never"], med["always"]
    split_key = [k for k in med if k.startswith("drop@")][0]
    inj_key = [k for k in med if k.startswith("inject@") and "late" not in k][0]
    damage = acc_never - acc_always
    rec = (med[split_key] - acc_always) / max(damage, 1e-9)
    frac_split = (med[inj_key] - acc_always) / max(damage, 1e-9)
    frac_late = (med["inject@late"] - acc_always) / max(damage, 1e-9)
    R1 = damage >= floors["damage"]; R2 = rec >= 0.7; R3 = frac_late >= 0.9
    return dict(
        pilot=name, acc={k: round(v, 4) for k, v in med.items()},
        damage=round(damage, 4), recovery_drop_at_split=round(rec, 3),
        frac_inject_at_split=round(frac_split, 3), frac_inject_late=round(frac_late, 3),
        R1=dict(floor=floors["damage"], pass_=bool(R1)),
        R2=dict(floor=0.7, pass_=bool(R2)), R3=dict(floor=0.9, pass_=bool(R3)),
        all_pass=bool(R1 and R2 and R3),
    )

def main():
    out = {}
    # pilot17: preregistered floors damage>=0.01, rec>=0.7, frac_late>=0.9
    p17 = os.path.join(HERE, "pilot17_realconflict_full_out.json")
    l17 = os.path.join(HERE, "pilot17_full.log")
    arms17 = ["never", "always", "drop@120", "inject@120", "inject@late"]
    if os.path.exists(p17):
        d = from_json(p17)
        out["pilot17"] = dict(verdict("pilot17", {k: v for k, v in d["acc"].items()}, {"damage": 0.01}),
                              source="json", nseeds=len(d["seeds"]))
    else:
        r = from_log(l17, arms17)
        if r:
            med, nseeds, rows = r
            out["pilot17"] = dict(verdict("pilot17", med, {"damage": 0.01}),
                                  source="log-partial", nseeds=nseeds)
    # pilot18: floors damage>=0.005 (small block), rec>=0.7, frac_late>=0.9
    p18 = os.path.join(HERE, "pilot18_smallconflict_full_out.json")
    l18 = os.path.join(HERE, "pilot18_full.log")
    if os.path.exists(p18):
        d = from_json(p18)
        out["pilot18"] = dict(verdict("pilot18", {k: v for k, v in d["acc"].items()}, {"damage": 0.005}),
                              source="json", nseeds=len(d["seeds"]))
    else:
        r = from_log(l18, arms17)
        if r:
            med, nseeds, rows = r
            out["pilot18"] = dict(verdict("pilot18", med, {"damage": 0.005}),
                                  source="log-partial", nseeds=nseeds)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "pilot1718_verdict.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
