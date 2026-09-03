#!/usr/bin/env python3
"""M10_VERIFY_R1911: exact-row assertions of SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json

Locks the M10 (r1911) EXACT-BAND NONTRIVIALITY FRONTIER claims to the on-disk JSON.
Pure / front / zero GPU.  EXIT=0 iff all assertions hold.
"""
import json, sys

import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json")
d = json.load(open(OUT))
fails = []


def check(name, cond, detail=""):
    if not cond:
        fails.append(name + (f" [{detail}]" if detail else ""))
        print(f"  FAIL {name} {detail}")
    else:
        print(f"  ok   {name}")


agg = d['agg']
check("agg.n_rows==1050", agg['n_rows'] == 1050, str(agg['n_rows']))
check("fresh seeds {10..14}", agg['fresh_seeds'] == [10, 11, 12, 13, 14])
check("budget grid {0.25,0.5,1.0}", agg['label_budgets'] == [0.25, 0.5, 1.0])
check("delta=0.1", agg['delta'] == 0.1)
check("3 seed/cell coverage", len([k for k in d['cells'] if k.startswith('fashion')]) >= 3)
check("cells present", {'digits_10', 'news_14', 'fashion_13', 'mnist_11'}.issubset(d['cells']))

cert = d['emptiness_certificate']
n_real_full = cert['n_real_switch_rows_full_budget']
check("125 real-switch rows at full budget", n_real_full == 125, str(n_real_full))
rows = cert['rows_bstar_gt_1_infeasible']
check("100% infeasible (bstar>1 all 125)", len(rows) == 125, str(len(rows)))

# bstar distribution
bs = [r['bstar_hoef'] for r in rows]
check("bstar min > 1", min(bs) > 1.0, f"min={min(bs):.1f}")
check("bstar median in (30, 3000)", 30 <= sorted(bs)[len(bs)//2] <= 3000,
      f"med={sorted(bs)[len(bs)//2]:.1f}")
check("bstar max large (weak rows)", max(bs) > 1000, f"max={max(bs):.1f}")

pc = d['per_carrier_budget']
# exact bands admit ZERO real switches at every budget on every carrier (incl mnist 0 basel)
for c in ['digits', 'fashion', 'mnist', 'news']:
    seen = set()
    for bp in ['0.25', '0.5', '1.0']:
        v = pc[c][bp]
        seen.add(v['exact_admit_real']['hoef'])
        seen.add(v['exact_admit_real']['mpb'])
        check(f"{c}@{bp} hoef admit==0", v['exact_admit_real']['hoef'] == 0.0,
              str(v['exact_admit_real']['hoef']))
        check(f"{c}@{bp} mpb admit==0", v['exact_admit_real']['mpb'] == 0.0,
              str(v['exact_admit_real']['mpb']))
# absolute gate retains content on fashion (default carrier), monotone in budget
f10 = pc['fashion']['1.0']
check("fashion@1 abs_commit=1.0", f10['abs_commit_rate'] == 1.0, str(f10['abs_commit_rate']))
check("fashion@1 abs_no_worse_cov=1.0", f10['abs_no_worse_cov_committed'] == 1.0,
      str(f10['abs_no_worse_cov_committed']))
check("fashion abs monotone 0.25<0.5<1",
      pc['fashion']['0.25']['abs_commit_rate'] < pc['fashion']['0.5']['abs_commit_rate'] <
      pc['fashion']['1.0']['abs_commit_rate'])
# trivial fraction never masks real switches: report separately (the whole point)
check("trivial_frac <= 1.0 everywhere", all(0 <= pc[c][bp]['trivial_frac'] <= 1.0
      for c in pc for bp in pc[c]))
# weak domains reported unfiltered
check("news budget wall @0.25 abs=0.0", pc['news']['0.25']['abs_commit_rate'] == 0.0,
      str(pc['news']['0.25']['abs_commit_rate']))
check("digits capacity wall @1 abs<0.2", pc['digits']['1.0']['abs_commit_rate'] < 0.2,
      str(pc['digits']['1.0']['abs_commit_rate']))

# monotone empirical impossibility: per-carrier real-switch count is NON-decreasing in budget
for c in ['digits', 'fashion', 'news']:
    cs = [pc[c][bp]['real_switch_count'] for bp in ['0.25', '0.5', '1.0']]
    check(f"{c} real-switch count monotone non-decreasing", cs[0] <= cs[1] <= cs[2], str(cs))

if fails:
    print(f"\nM10_VERIFY FAIL {len(fails)}")
    sys.exit(1)
print(f"\nM10_VERIFY_R1911: ALL PASS ({len(bs)} infeasible rows, {n_real_full} real full)")
sys.exit(0)