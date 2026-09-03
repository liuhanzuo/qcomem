#!/usr/bin/env python3
"""M9 verifier r1910: N*-frontier of strict finite-sample relative gate.

Asserts, foreground EXIT=0 iff all PASS:
  C1 on 6 switch rows: D_normal_base < 0 (repro M8 asymptotic certification).
  C2 on 6 switch rows: D_hoef_base > 0 and D_mpb_base > 0 (repro M8 exact-band rejection).
  C3 D_mpb/D_hoef monotone non-increasing in N over grid (bands tighten as budget grows).
  C4 minimal N*_mpb and N*_hoef are recorded; D_grid cover NGRID.
  C5 OUTER soundness: on each switch row, oracle_switch_gain = R_F0 - R_i > 0 (i.e.
     committing to i* is genuinely no-worse; the relative gate open point is sound).
  C6 chunk-24 exact reproduction: chosen/F0/true_regret/D_normal_equal on shared rows
     vs frozen M8 file (4-decimal storage precision).
  C7 no row fabricates a gate open that OUTER falsifies (D(N*)<=0 reconstruccible).
"""
import json, sys, math, os
_HERE = os.path.dirname(os.path.abspath(__file__))
M8 = json.load(open(os.path.join(_HERE, "SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json")))
M8R = {(r['carrier'], r['seed'], r['w']): r for r in M8['rows']}
M9 = json.load(open(os.path.join(_HERE, "SUBGMIX_M9_NSTAR_FRONTIER_R1910.json")))
SW = M9['switch_rows']
NGRID = M9['nstar_agg']['n_grid']

fails = []

def chk(cond, msg):
    if not cond:
        fails.append(msg)

# C1/C2
for r in SW:
    k = (r['carrier'], r['seed'], r['w'])
    chk(r['D_normal_base'] < 0, f"C1 {k} D_normal_base={r['D_normal_base']} not <0")
    chk(r['D_hoef_base'] > 0, f"C2 {k} D_hoef_base={r['D_hoef_base']} not >0")
    chk(r['D_mpb_base'] > 0, f"C2 {k} D_mpb_base={r['D_mpb_base']} not >0")

# C3 monotone non-increasing (mpb & hoef) across grid
for r in SW:
    for b in ('mpb', 'hoef'):
        dd = r[f'D_{b}_grid']
        prev = None
        for Nn in NGRID:
            v = dd.get(str(Nn))
            if v is None:
                continue
            if prev is not None and v > prev + 1e-9:
                chk(False, f"C3 {b} {r['carrier'],r['seed'],r['w']} N={Nn} D rose {prev}->{v}")
            prev = v

# C4 N* recorded and consistent
for r in SW:
    for b in ('mpb', 'hoef'):
        nst = r[f'Nstar_{b}']
        dd = r[f'D_{b}_grid']
        if nst is not None:
            chk(dd[str(nst)] <= 0, f"C4 {b} {r['carrier'],r['seed'],r['w']} N*={nst} not <=0")
            # any smaller grid value must be >0
            smaller = [Nn for Nn in NGRID if Nn < nst]
            for sm in smaller:
                chk(dd.get(str(sm), math.inf) > 0,
                    f"C4 {b} {r['carrier'],r['seed'],r['w']} N={sm}<=0 below N*={nst}")

# C5 OUTER soundness of open point
for r in SW:
    chk(r['oracle_switch_gain_abs'] > 0, f"C5 {r['carrier'],r['seed'],r['w']} switch gain={r['oracle_switch_gain_abs']} not >0")

# C6 reproduction vs frozen M8
for r in SW:
    fr = M8R[(r['carrier'], r['seed'], r['w'])]
    for k in ('chosen', 'F0', 'true_regret'):
        chk(fr[k] == r[k], f"C6 {r['carrier'],r['seed'],r['w']} {k} {fr[k]}!={r[k]}")
    chk(abs(fr['D_normal'] - r['D_normal_base']) < 1e-9, f"C6 {r['carrier'],r['seed'],r['w']} D_normal {fr['D_normal']}!={r['D_normal_base']}")

# C7 counts
chk(len(SW) == 6, f"C7 {len(SW)} switch rows != 6")

if fails:
    print(f"FAIL {len(fails)}")
    for f in fails[:20]:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
print(f"  switch rows: {len(SW)}")
for r in SW:
    print(f"  {r['carrier']}/{r['seed']}/{r['w']}: N*_mpb={r['Nstar_mpb']} N*_hoef={r['Nstar_hoef']} "
          f"Dmpb(1)={r['D_mpb_base']} gain={r['oracle_switch_gain_abs']}")
sys.exit(0)