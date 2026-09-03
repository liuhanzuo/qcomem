"""M8 r1909 verifier: lock the manuscript-cited finite-sample relative-gate numbers.

Asserts against the freshly produced SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json the exact
figures used in Sec.~app:tau (M6 continued) and the Conclusion/Limitations edits:
  * normal band reproduces frozen M6 (commit_rate 0.6629).
  * exact bands (hoef, mpb) commit_rate 0.6457, no_worse cov 1.0.
  * MECHANISM: 226 rows have i*==F0, ALL exact-band commits have i*==F0 (trivial); the 6
    non-F0 switch proposals are committed by NORMAL but rejected by hoef/mpb.
Front: exit 0 == all PASS.  PROJECT A2_SAFE_MODEL_RANKING_SUBGROUP_MIX. ROUND r1909.
"""
import json, numpy as np, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json")
d = json.load(open(OUT))
agg = d['agg']; rows = d['rows']
assert agg['n_rows'] == 350 and agg['seeds'] == [0, 1, 2, 3, 4], agg
P = []

def chk(name, cond, val=None):
    global P
    P.append((name, bool(cond), val))

# 1. normal == frozen M6
a = agg['normal']
chk('normal_commit_rate_eq_m6_0.6629', abs(a['commit_rate'] - 0.6629) < 1e-4, a['commit_rate'])
chk('normal_sq_cov_1.0', a['sq_no_worse_cov_upgraded'] == 1.0)
chk('normal_sq_max_0', a['sq_max_upgraded'] == 0.0)
chk('normal_or_max_committed_0.0546', abs(a['or_max_committed'] - 0.0546) < 1e-4, a['or_max_committed'])

# 2. exact bands one value each; claim = same 0.6457
for b in ['hoef', 'mpb']:
    x = agg[b]
    chk(f'{b}_commit_0.6457', abs(x['commit_rate'] - 0.6457) < 1e-4, x['commit_rate'])
    chk(f'{b}_sq_cov_1.0_upgraded', x['sq_no_worse_cov_upgraded'] == 1.0)
    chk(f'{b}_sq_max_0', x['sq_max_upgraded'] == 0.0)

# 3. mechanism
n_f0 = sum(1 for r in rows if r['chosen'] == r['F0'])
chk('n_f0_is_226', n_f0 == 226, n_f0)
for b in ['hoef', 'mpb']:
    comm = [r for r in rows if r['commit_' + b]]
    all_f0 = all(r['chosen'] == r['F0'] for r in comm)
    chk(f'{b}_all_commits_are_F0_trivial', all_f0, len(comm))
    # every switch row rejected
    sw = [r for r in rows if r['chosen'] != r['F0'] and r['commit_' + b]]
    chk(f'{b}_zero_switch_commits', len(sw) == 0, len(sw))
# normal: exactly 6 switch commits
sw_norm = [r for r in rows if r['chosen'] != r['F0'] and r['commit_normal']]
chk('normal_six_switch_commits', len(sw_norm) == 6, len(sw_norm))
# soundness of those 6 on OUTER (REG_sq<=0)
chk('normal_switches_all_sound', all(r['REG_sq_normal'] <= 1e-9 for r in sw_norm))

fails = [x for x in P if not x[1]]
print(f"M8_VERIFY_R1909: {len(P)-len(fails)}/{len(P)} PASS")
for f in fails:
    print("  FAIL:", f)
sys.exit(0 if not fails else 1)