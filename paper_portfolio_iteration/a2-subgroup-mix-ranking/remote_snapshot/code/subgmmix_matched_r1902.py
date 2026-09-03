"""r1902 matched-seed aggregation: recompute the headline endpoints of M2, M2.5-paired,
and M2.5-MPB on the SAME 3 seeds (0,1,2), the identical data splits, calibration points,
m3/w grid, budget, certificate and carriers that M2.5 used.

WHY THIS FILE EXISTS (MGR instruction ea0889891916): the paper headline juxtaposed M2's
5-seed committed_rate 0.269 against M2.5's 3-seed 0.495/0.257.  Because every model is
trained deterministically from random_state=seed and all splits derive from the same seed,
shrinking M2 to seeds {0,1,2} yields a strict, retraining-free matched baseline: the same
trained models, same calibration pools, same w_grid, same delta/gate.  This module
re-aggregates the frozen JSONs on that matched seed set.  It reads only frozen evidence
and writes nothing; the check runner asserts the paper's headline table numbers against
this matched set.

The 5-seed M2 numbers are preserved in the frozen JSON and re-labeled in the paper as an
independent robustness view, NOT as the direct baseline of the 3-seed methods.

KEYDERIVED (matched, seeds {0,1,2}, tau=0.04, delta=0.10):
  M2          committed_rate 0.2667  cov 1.0  comm_mean 0.00001 comm_max 0.00050
                            abst_mean 0.00442 abst_max 0.05460 no_gate 0.00324
  M2.5-paired committed_rate 0.4952  cov 1.0
  M2.5-MPB    committed_rate 0.2571  cov 1.0
  M2.5-Hoeffd committed_rate 0.0905  cov 1.0
Exact finite-sample order (matched): MPB 0.2571 < M2 0.2667 (still slightly lower; keep W1).
"""
import json, os, numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
TAU = 0.04
MATCHED = {0, 1, 2}
CARRIERS = ['digits', 'fashion', 'mnist', 'news']


def _load(fn):
    with open(os.path.join(ROOT, fn)) as f:
        return json.load(f)


def matched_rows(obj, committed_test):
    """Return (rows, agg) for the matched seed set. committed_test: fn(r)->bool."""
    rows = [r for r in obj['rows'] if r['seed'] in MATCHED]
    n = len(rows)
    comm = [r for r in rows if committed_test(r)]
    abst = [r for r in rows if not committed_test(r)]
    cm = [r['true_regret'] for r in comm]
    am = [r['true_regret'] for r in abst]
    allg = [r['true_regret'] for r in rows]
    cov = float(np.mean([x <= TAU + 1e-9 for x in cm])) if cm else float('nan')
    agg = {
        'n_rows': n, 'n_committed': len(comm), 'committed_rate': round(len(comm) / n, 4),
        'cert_coverage': round(cov, 4),
        'comm_mean_regret': round(float(np.mean(cm)), 5) if cm else None,
        'comm_max_regret': round(float(max(cm)), 5) if cm else None,
        'abst_mean_regret': round(float(np.mean(am)), 5) if am else None,
        'abst_max_regret': round(float(max(am)), 5) if am else None,
        'no_gate_mean_regret': round(float(np.mean(allg)), 5),
        'tau': TAU, 'matched_seeds': sorted(MATCHED),
    }
    return rows, agg


def m2_matched():
    return matched_rows(_load('SUBGMIX_M2_GATE_R1884.json'), lambda r: r['committed'])


def m25_paired_matched():
    return matched_rows(_load('SUBGMIX_M25_PAIRED_R1885.json'), lambda r: r['committed'])


def m25_mpb_matched():
    return matched_rows(_load('SUBGMIX_M25_PAIRED_R1885.json'), lambda r: r['UB_paired_mpb'] <= TAU)


def m25_hoef_matched():
    return matched_rows(_load('SUBGMIX_M25_PAIRED_R1885.json'), lambda r: r['UB_paired_hoef'] <= TAU)


def per_carrier(rows, committed_test):
    out = {}
    for c in CARRIERS:
        rs = [r for r in rows if r['carrier'] == c]
        n = len(rs)
        cm = [r for r in rs if committed_test(r)]
        abst = [r for r in rs if not committed_test(r)]
        cg = [r['true_regret'] for r in cm]
        ag = [r['true_regret'] for r in abst]
        out[c] = {'n': n, 'committed': len(cm),
                  'rate': round(len(cm) / n, 3) if n else None,
                  'comm_mean': round(float(np.mean(cg)), 5) if cg else None,
                  'comm_max': round(float(max(cg)), 5) if cg else None,
                  'abst_mean': round(float(np.mean(ag)), 5) if ag else None,
                  'abst_max': round(float(max(ag)), 5) if ag else None}
    return out


if __name__ == '__main__':
    for lbl, rows, agg, test in [
        ('M2', *m2_matched(), lambda r: r['committed']),
        ('M2.5-paired', *m25_paired_matched(), lambda r: r['committed']),
        ('M2.5-MPB', *m25_mpb_matched(), lambda r: r['UB_paired_mpb'] <= TAU),
        ('M2.5-Hoeffding', *m25_hoef_matched(), lambda r: r['UB_paired_hoef'] <= TAU),
    ]:
        print(f"=== {lbl} (matched seeds {MATCHED}) ===")
        print('  agg:', json.dumps(agg, indent=2))
        print('  per_carrier:', json.dumps(per_carrier(rows, test), indent=2))