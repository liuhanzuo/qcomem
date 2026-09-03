#!python3
"""M12 (r1915) verifier: re-derive every M12 headline from SUBGMIX_M12_FRESH5_BUDGET_R1915.json.

Reads only the frozen M12 JSON; writes nothing. EXIT 0 iff all checks pass.
Locks the r1915 headline claims:
  (1) exact RELATIVE band (Hoeffding AND MPB) admits ZERO genuine switches (i*!=F0 committed)
      on every (carrier, frac, rule) cell across fresh seeds {10..14} -- the M10/M11 whole-axis
      emptiness re-confirmed at the M3/M3.5 budget totals, by real without-replacement sampling.
  (2) exact ABSOLUTE gate (M2.5) is sound (abs_truecov==1.0 on every cell that commits).
  (3) absolute content is monotone in budget for the strong carriers.
  (4) digits/news abscissa commit 0 (capacity/budget wall).
"""
import json, sys, numpy as np, os
ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep
d = json.load(open(ROOT + 'SUBGMIX_M12_FRESH5_BUDGET_R1915.json'))
pc = d['per_carrier']
PASS = 0; FAIL = []

def chk(name, got, want, tol=1e-6):
    global PASS
    ok = (want is None) or (abs(float(got) - float(want)) <= tol)
    if ok: PASS += 1
    else: FAIL.append(f"{name}: got={got} want={want}")

def chk_near(name, got, lo, hi):
    global PASS
    ok = (got is None) or (lo - 1e-9 <= float(got) <= hi + 1e-9)
    if ok: PASS += 1
    else: FAIL.append(f"{name}: got={got} outside [{lo},{hi}]")

RULES = ['uniform', 'neyman', 'sens', 'widthgreedy', 'convexminimax']
FRACS = ['0.5', '0.65', '0.8', '0.95']
CARS = ['digits', 'fashion', 'mnist', 'news']

# (1) exact-relative admits zero genuine switches on every cell
for name in CARS:
    for f in FRACS:
        for rule in RULES:
            g = pc[name][f][rule]
            chk(f"{name}/{f}/{rule}/rel_hoef_admit", g.get('rel_real_admit_hoef'), 0.0)
            chk(f"{name}/{f}/{rule}/rel_mpb_admit", g.get('rel_real_admit_mpb'), 0.0)

# (2) absolute gate sound on every committing cell
sound_cells = 0
for name in CARS:
    for f in FRACS:
        for rule in RULES:
            g = pc[name][f][rule]
            if g['abs_commit'] and g['abs_commit'] > 0:
                sound_cells += 1
                chk_near(f"{name}/{f}/{rule}/abs_truecov", g['abs_truecov'], 1.0, 1.0)
assert sound_cells > 0, "no committing cell -> cannot verify soundness"
chk("n_sound_cells >= 20 (many cells commit)", sound_cells >= 20, True)

# (3) monotone absolute content in budget
chk("mnist/uniform pi0.5", pc['mnist']['0.5']['uniform']['abs_commit'], 0.7733, 1e-3)
chk("mnist/uniform pi0.8", pc['mnist']['0.8']['uniform']['abs_commit'], 1.0000, 1e-3)
chk("mnist/uniform pi0.95", pc['mnist']['0.95']['uniform']['abs_commit'], 1.0000, 1e-3)
fh = [pc['fashion']['0.5']['uniform']['abs_commit'],
      pc['fashion']['0.8']['uniform']['abs_commit'],
      pc['fashion']['0.95']['uniform']['abs_commit']]
chk("fashion/uniform monotone 0.5<=0.8<=0.95", (fh[0] <= fh[1] + 1e-9) and (fh[1] <= fh[2] + 1e-9), True)
chk("fashion/uniform pi0.95 abs_cr", fh[2], 0.2000, 1e-3)
chk("fashion/uniform pi0.95 abs_gain_max", pc['fashion']['0.95']['uniform'].get('abs_gain_max'), 0.07525, 2e-3)

# (4) capacity/budget wall on digits & news
for f in FRACS:
    for rule in RULES:
        chk(f"digits/{f}/{rule}/abs_commit wall", pc['digits'][f][rule]['abs_commit'], 0.0)
        chk(f"news/{f}/{rule}/abs_commit wall", pc['news'][f][rule]['abs_commit'], 0.0)

# sanity: real-switch counts present (non-trivial structure exists to certify)
tot_real = sum(pc[name][f][rule]['real_switch_count']
               for name in ['fashion', 'news', 'digits'] for f in FRACS for rule in RULES)
chk("real-switch rows exist (>= 1000)", tot_real >= 1000, True)

print(f"PASS {PASS}, FAIL {len(FAIL)}")
if FAIL:
    for f in FAIL[:40]:
        print("  FAIL:", f)
    sys.exit(1)
sys.exit(0)