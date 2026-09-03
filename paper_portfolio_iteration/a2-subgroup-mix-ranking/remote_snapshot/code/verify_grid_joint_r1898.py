"""r1898: verify the paired-difference certificate is SIMULTANEOUS over the whole
mixture simplex (not merely per-w pointwise / grid-joint), via a dense OFF-GRID
Dirichlet scan on the same frozen m3_cache.

Construction being verified: UCB[(i,j)][g] are w-INDEPENDENT one-sided MPB/paired
bounds, each at level  dcell = DELTA / (M*(M-1)*G)  (Bonferroni over ordered pairs
x groups ONLY --- no factor of |grid|).  Thus the single joint event
    E = { for all ordered (i,j), all g :  r_{i,g}-r_{j,g} <= UCB[(i,j)][g] }
has P(E) >= 1-DELTA, and since the bound
    UB(i,w) = max_{j!=i} sum_g w_g UCB[(i,j)][g]
is a deterministic linear-in-w combination of UCBs, under E  it validates
    regret(i,w) = max_j sum_g w_g (r_{i,g}-r_{j,g}) <= UB(i,w)
     SIMULTANEOUSLY for EVERY simplex point w and EVERY CAL-selected i*.
This is the standard "simultaneous best comparison" (MCB) argument: because the
selection i*(w) is data-dependent but the joint event covers ALL ordered pairs,
conditioning on the realized i* is safe.  Hence the disclosed limitation "certificate
per-w pointwise; grid-joint TBD" is over-conservative: the guarantee already holds
jointly over the continuous simplex.

Verification (soundness, empirical, OUTER): fix the exact reveal sets used by the
M3 uniform allocation at each frac (same frozen m3_cache, same seed, same delta/tau),
build UCB once, then scan (a) the original grid AND (b) a dense OFF-GRID Dirichlet
w-sample (1000/carrier*seed*frac) NOT on any grid line.  Check cert coverage on
committed cells is still 1.0 and report committed_rate for both sets.  If the
off-grid coverage stays 1.0 while committing a comparable rate, the certificate is
simultaneous over the simplex, not a finite-grid artifact.

PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX  ROUND r1898  Pure CPU / front / zero GPU.
"""
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code'))
from subgmmix_minimax_r1895 import (load_art, grid_from_recs, dcell,
                                    build_ucb_mpb, select_cert,
                                    DELTA, TAU, SEEDS, CARRIERS, FRACS)

OUT = os.path.join(ROOT, 'results', 'VERIFY_GRID_JOINT_R1898.json')
N_OFFGRID = 1000          # dense off-grid Dirichlet w per (carrier,seed,frac)
RNG_OFFSET = 777777       # distinct seed family from the CAL split seeds


def dirichlet_w(G, rng):
    w = rng.dirichlet(np.ones(G))
    return {g: float(w[g]) for g in range(G)}


def run_carrier(name, seed):
    errs, fit_err, outer, yf, yc, Mnames = load_art(name, seed)
    G = int(yc.max() + 1)
    grid = grid_from_recs(name)
    avail = {g: int((yc == g).sum()) for g in range(G)}
    # pretend uniform reveal at each frac (same as M3 uniform column); to keep the
    # revealed set independent of mixture w, fix orders once per seed.
    orders = {g: np.random.RandomState(1000 + seed * 10 + g).permutation(
        np.where(yc == g)[0]).tolist() for g in range(G)}
    rng = np.random.RandomState(RNG_OFFSET + seed * 100 + 0)
    res_frac = {}
    for frac in FRACS:
        R = int(round(frac * len(yc)))
        R = min(R, sum(avail.values()))
        n = {g: max(1, int(round(R / G))) for g in range(G)}
        rem = R - sum(n.values())
        for k in range(rem):
            n[list(range(G))[k % G]] += 1
        rev = np.zeros(len(yc), bool)
        for g in range(G):
            if n[g] > 0:
                rev[orders[g][:n[g]]] = True
        dc = dcell(len(Mnames), G)
        UCB = build_ucb_mpb(errs, yc, rev, Mnames, G, dc)
        # on-grid cells
        og = []
        for gp in grid:
            w = gp['w']
            i, ub = select_cert(errs, yc, rev, UCB, Mnames, G, w)
            trueR = {m: sum(w[g] * outer[m][g] for g in range(G)) for m in Mnames}
            reg = trueR[i] - min(trueR.values())
            og.append({'w': gp['name'], 'chosen': i, 'UB': float(ub),
                       'true_regret': float(reg), 'committed': bool(ub <= TAU)})
        # dense off-grid Dirichlet scan (same UCB family, NO extra per-w split)
        off = []
        for _ in range(N_OFFGRID):
            w = dirichlet_w(G, rng)
            i, ub = select_cert(errs, yc, rev, UCB, Mnames, G, w)
            trueR = {m: sum(w[g] * outer[m][g] for g in range(G)) for m in Mnames}
            reg = trueR[i] - min(trueR.values())
            off.append({'chosen': i, 'UB': float(ub), 'true_regret': float(reg),
                        'committed': bool(ub <= TAU)})
        res_frac[frac] = {'on_grid': og, 'off_grid': off, 'R': int(R), 'G': G,
                          'M': len(Mnames), 'dcell': dc}
    return G, grid, res_frac, len(Mnames)


def summarize(cells):
    """cells: list of dicts with 'committed' and 'true_regret'."""
    comm = [c for c in cells if c['committed']]
    if comm:
        cov = float(np.mean([c['true_regret'] <= TAU + 1e-9 for c in comm]))
        mx = float(np.max([c['true_regret'] for c in comm]))
        mnr = float(np.mean([c['true_regret'] for c in comm]))
    else:
        cov, mx, mnr = None, None, None
    return {'n_cells': len(cells), 'n_committed': len(comm),
            'committed_rate': round(float(len(comm) / len(cells)), 4),
            'cert_coverage': round(float(cov), 4) if cov is not None else None,
            'comm_max_regret': round(float(mx), 4) if mx is not None else None,
            'comm_mean_regret': round(float(mnr), 4) if mnr is not None else None}


def main():
    t0 = time.time()
    ACC = {}
    for name in CARRIERS:
        for seed in SEEDS:
            G, grid, res_frac, M = run_carrier(name, seed)
            for frac, rr in res_frac.items():
                so = summarize(rr['on_grid'])
                sf = summarize(rr['off_grid'])
                ACC.setdefault((name, frac), []).append({
                    'seed': seed, 'on_grid': so, 'off_grid': sf,
                    'dcell': rr['dcell'], 'G': G, 'M': M})
    agg = []
    for (name, frac), vals in sorted(ACC.items()):
        og_cr = np.mean([v['on_grid']['committed_rate'] for v in vals])
        og_cv = np.mean([v['on_grid']['cert_coverage'] for v in vals if v['on_grid']['cert_coverage'] is not None])
        of_cr = np.mean([v['off_grid']['committed_rate'] for v in vals])
        of_cv = np.mean([v['off_grid']['cert_coverage'] for v in vals if v['off_grid']['cert_coverage'] is not None])
        total_off = sum(v['off_grid']['n_committed'] for v in vals)
        agg.append({'carrier': name, 'frac': frac, 'G': vals[0]['G'], 'M': vals[0]['M'],
                    'n_seeds': len(vals),
                    'on_grid_committed_rate': round(float(og_cr), 3),
                    'on_grid_cert_coverage': round(float(og_cv), 4),
                    'off_grid_committed_rate': round(float(of_cr), 3),
                    'off_grid_cert_coverage': round(float(of_cv), 4),
                    'off_grid_total_committed': int(total_off)})
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1898',
           'kind': 'simplex_simultaneous_certificate_dense_offgrid_scan',
           'tau': TAU, 'delta': DELTA, 'n_offgrid_per_cell': N_OFFGRID,
           'claim': 'paired-diff UCBs are w-independent (Bonferroni over ordered pairs x '
                    'groups only), so the joint event covers the CONTINUOUS simplex; '
                    'off-grid cert-coverage==1.0 confirms not a finite-grid artifact.',
           'agg': agg, 'runtime_s': round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"{'carrier':8}{'frac':>6} {'on.CR/cov':>14} {'off.CR/cov':>16} {'off.comm':>8}")
    for a in agg:
        oC = a['on_grid_cert_coverage']; fC = a['off_grid_cert_coverage']
        print(f"{a['carrier']:8}{a['frac']:>6} {a['on_grid_committed_rate']:>7.3f}/{oC:.3f} "
              f"{a['off_grid_committed_rate']:>8.3f}/{fC:.3f} {a['off_grid_total_committed']:>8}")
    print('saved', OUT, 'runtime', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()