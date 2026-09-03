"""M2.5 paired-difference certificate, full 5-seed set {0,1,2,3,4}.

r1903: closes the honest gap that M2 (5-seed 0.269) and M2.5 (3-seed 0.495/0.257) were
NOT on the same seed set.  Every model trains deterministically from random_state=seed
and all splits derive from the same seed, so running the same M2.5 paired certificate on
seeds {0,1,2,3,4} (identical carriers, w_grid, delta, gate, budget) yields a STRICT
5-seed matched M2.5 to compare against the frozen 5-seed M2.

Reuses the frozen run_carrier from subgmmix_m25_paired_r1885.py verbatim (no logic change);
only the seed loop is widened.  Writes a NEW frozen JSON SUBGMIX_M25_PAIRED_R1885_5SEED.json
(3-seed r1885 file is left untouched as an archived snapshot).
PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX. ROUND: r1903. Pure CPU / front / zero GPU.
"""
import json, time, numpy as np, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from subgmmix_m25_paired_r1885 import run_carrier

OUT = "subgroup_mix_ranking/results/SUBGMIX_M25_PAIRED_R1885_5SEED.json"
TAU = 0.04
SEEDS = [0, 1, 2, 3, 4]
DELTA = 0.10
CARRIERS = ['digits', 'fashion', 'mnist', 'news']


def main():
    t0 = time.time(); all_rows = []
    for name in CARRIERS:
        for seed in SEEDS:
            all_rows.extend(run_carrier(name, seed))
    n = len(all_rows)
    n_comm = sum(1 for r in all_rows if r['committed'])
    comm_reg = [r['true_regret'] for r in all_rows if r['committed']]
    abs_reg = [r['true_regret'] for r in all_rows if not r['committed']]
    cov = np.mean([r <= TAU + 1e-9 for r in comm_reg]) if comm_reg else float('nan')

    n_comm_hoef = sum(1 for r in all_rows if r['UB_paired_hoef'] <= TAU)
    comm_hoef_reg = [r['true_regret'] for r in all_rows if r['UB_paired_hoef'] <= TAU]
    cov_hoef = np.mean([r <= TAU + 1e-9 for r in comm_hoef_reg]) if comm_hoef_reg else float('nan')

    rows_mpb = [r for r in all_rows if r['UB_paired_mpb'] <= TAU]
    reg_mpb = [r['true_regret'] for r in rows_mpb]
    cov_mpb = np.mean([r <= TAU + 1e-9 for r in reg_mpb]) if reg_mpb else float('nan')

    agg = {'n_rows': n, 'seeds': SEEDS, 'tau': TAU, 'delta': DELTA,
           'committed_rate_pair': round(n_comm / n, 3),
           'cert_cov_pair': round(float(cov), 4),
           'comm_mean_regret': round(float(np.mean(comm_reg)), 4) if comm_reg else None,
           'comm_max_regret': round(float(np.max(comm_reg)), 4) if comm_reg else None,
           'abst_mean_reg': round(float(np.mean(abs_reg)), 4) if abs_reg else None,
           'abst_max_reg': round(float(np.max(abs_reg)), 4) if abs_reg else None,
           'committed_rate_hoef': round(n_comm_hoef / n, 3),
           'cert_cov_hoef': round(float(cov_hoef), 4),
           'committed_rate_mpb': round(len(rows_mpb) / n, 3),
           'cert_cov_mpb': round(float(cov_mpb), 4),
           'runtime_s': round(time.time() - t0, 1)}
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1903',
           'note': '5-seed matched extension of r1885 M2.5 paired', 'agg': agg, 'rows': all_rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(agg, indent=2)); print('saved', OUT)
    sys.exit(0)


if __name__ == '__main__':
    main()