#!/usr/bin/env python3
"""A11 r484: the critical-radius conservation curve of Proposition prop:tv,
across all three carriers (OMR N=32 shards 0/1, RLVE N=8, OpenR1 M=2).

tau*(alpha) = max{R : V(R) <= alpha} is computed EXACTLY on a dense grid
alpha in [1e-3, 0.20] (0.0005 step; for OpenR1 the closed form is exact at
every alpha, not a grid approximation). Zero new data, zero TEST re-reads:
the g-profiles are re-derived in-process from the same frozen FIT priors and
the same DP/certificate machinery as r481 (fit_cal_test_r469.py exec-prefix),
r482 (rlve_n8_r474.py module import) and r483 (analytic 3-atom certs).

Pre-registered predictions (written BEFORE first run; P1 self-corrected
after the first run, disclosed in the result JSON):
  P1: within each alpha-interval where the frozen rule (hence the whole
      g-vector) is fixed, V(R) is a fixed non-decreasing function, so
      tau*(alpha) = max{R : V(R) <= alpha} is NON-DECREASING in alpha;
      jumps (up or down) occur exactly at the alpha values where the rule
      changes (certificate thresholds). SELF-AUDIT CORRECTION: the first
      version of this docstring wrote "non-increasing" (wrong direction)
      and the first breakpoint scan keyed on g_max only, which MISSED
      rule changes that leave g_max unchanged (caught by two RLVE
      within-interval decreases at alpha .0075->.008 and .043->.0435:
      V(0.368) changes 0.00750->0.00856 with identical g_max, proving the
      g-vector changed). Breakpoints are now detected on the full g-vector.
  P2: OpenR1: tau*(a) = a/g_mid(a) - Hmid with g_mid=.25 (a>=.07852),
      .125 (.04598<=a<.07852), and tau*=1 (whole simplex) for a<.04598
      (never-stop rule, V==0). The curve is monotone non-decreasing within
      (0.07852, .20] and (.04598, .07852) and JUMPS DOWN at a=.07852
      (g_mid doubles) and UP to 1 at a=.04598 (rule never stops).
      Continuity conjecture to test: at the .07852 breakpoint from below,
      a/g_mid - Hmid = .07852/.125-.232 = .39616; from above
      .07852/.25-.232 = .08208 -- a downward jump of ~0.314. At .04598
      from above: .04598/.125-.232 = .13584; below: 1 -- upward jump.
  P3: The reference-level values reproduce r481/r482/r483 exactly:
      OMR s0 .144/.078/.074/.052, s1 .157/.082/.076/.024,
      RLVE .199/.551/.213/.449, OpenR1 .168/.168/1/1 at .10/.05/.02/.01.
  P4: The per-level zero-g alignment number printed in the paper (36-47% /
      74-87% / 76.8%) is monotone in tau* only ACROSS carriers, not within
      one carrier across levels (zero-g mass is itself alpha-dependent).
      We quantify: Spearman correlation across the 12 carrier x level cells
      between zero-g mass and tau* -- reported descriptively.

Outputs: tv_conservation_r484_result.json + fig_tau_conservation.png
(copied to the paper dir). matplotlib + stdlib + the frozen carrier modules.
No GPU/net. Readback: stdout + JSON.
"""
import json, os, random, importlib.util, hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
OUT = os.path.join(HERE, "tv_conservation_r484_result.json")
FIG = os.path.join(HERE, "fig_tau_conservation.png")
SEED = 20260815
AGRID_REF = [0.10, 0.05, 0.02, 0.01]
ALO, AHI, ASTEP = 0.001, 0.20, 0.0005


def alpha_grid():
    n = int(round((AHI - ALO) / ASTEP)) + 1
    return [ALO + i * ASTEP for i in range(n)]


# ---------------------------------------------------------------- OMR N=32
def load_r469_machinery():
    src_path = os.path.join(WS, "earlystop_drift_r469", "fit_cal_test_r469.py")
    src = open(src_path).read()
    ns = {}
    exec(compile(src[:src.index("def eb_ucb")], src_path, "exec"), ns)
    return ns["build_cert_table"], ns["dp_adaptive_flip"]


def load_omr_counts(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["problem", "pass_rate_72b_tir"]).to_pandas()
    t = t[t.pass_rate_72b_tir.notna() & (t.pass_rate_72b_tir != "n/a")].copy()
    t = t.drop_duplicates(subset=["problem"])
    return [int(round(p * 32)) for p in t.pass_rate_72b_tir.astype(float).tolist()]


def fit_omr_prior(Ks):
    rnd = random.Random(SEED)
    idx = list(range(len(Ks)))
    rnd.shuffle(idx)
    fit_idx = idx[:4000]
    H = [0.0] * 33
    for i in fit_idx:
        H[Ks[i]] += 1.0
    return [h / len(fit_idx) for h in H]


# ---------------------------------------------------------------- RLVE N=8
def load_rlve():
    spec = importlib.util.spec_from_file_location(
        "r474", os.path.join(WS, "earlystop_drift_r474", "rlve_n8_r474.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # main() is __name__-guarded
    mod.SHARDS = [os.path.join(WS, "earlystop_drift_r474",
                               f"rlve_p{i}.parquet") for i in range(6)]
    Ks, seqs, qenv = mod.load_data()
    rnd = random.Random(SEED)
    idx = list(range(len(Ks)))
    rnd.shuffle(idx)
    Hhat = mod.fit_prior(Ks, idx[:3000])
    cert = mod.build_cert_table(Hhat)
    return mod, Hhat, cert


# ------------------------------------------------------------- OpenR1 M=2
CERT_X0, CERT_X1 = 0.07852, 0.04598  # frozen r473 certs (fail/pass prefix)


def openr1_g(a):
    s0, s1 = CERT_X0 <= a, CERT_X1 <= a
    return [0.0, 0.25 * (0.5 * s0 + 0.5 * s1), 0.0]


def openr1_prior():
    with open(os.path.join(WS, "earlystop_drift_r473",
                           "openr1_m2_pilot_r473.json")) as f:
        return [float(h) for h in json.load(f)["prior_H_p0_p5_p1"]]


# --------------------------------------------------------- exact worst case
def worst_case(H, g, R):
    """Two-sided greedy LP optimum over simplex intersect L1 ball (2R)."""
    n = len(H) - 1
    base = sum(h * gv for h, gv in zip(H, g))
    if R <= 0:
        return base

    def side(desc):
        order = sorted(range(n + 1), key=lambda K: -g[K] if desc else g[K])
        rem, acc = R, 0.0
        for K in order:
            room = (1.0 - H[K]) if desc else H[K]
            take = min(room, rem)
            if take <= 0:
                continue
            acc += take * g[K]
            rem -= take
            if rem <= 1e-15:
                break
        return acc, R - rem

    add, moved_a = side(True)
    take_, moved_t = side(False)
    if min(moved_a, moved_t) < R - 1e-12:
        return None
    return base + add - take_


def tau_star(H, g, alpha):
    base = sum(h * gv for h, gv in zip(H, g))
    if base > alpha:
        return 0.0
    if max(g) <= alpha:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        v = worst_case(H, g, mid)
        if v is None or v > alpha:
            hi = mid
        else:
            lo = mid
    return lo


def g_profile_omr(build, dpflip, Hhat, a):
    # cert table identical for all a (built from Hhat); per-a DP flip
    cert = build(Hhat)
    g = [0.0] * 33
    for K in range(33):
        g[K] = dpflip(K, cert, a)[0]
    return g


def main():
    grid = alpha_grid()
    out = {"seed": SEED, "alpha_grid": {"lo": ALO, "hi": AHI, "step": ASTEP},
           "definition": "tau*(alpha)=max{R:V(R)<=alpha}, exact two-sided "
                         "greedy LP per alpha; OpenR1 via analytic g profile",
           "reference_alphas": AGRID_REF}

    build, dpflip = load_r469_machinery()
    curves = {}

    # --- OMR shard0 / shard1: cert table depends only on Hhat, not alpha;
    # build it ONCE per shard and reuse across the grid (200x33 DP calls).
    for cname, path in (("omr_shard0", "earlystop_drift_r467/cot_shard0.parquet"),
                        ("omr_shard1", "earlystop_drift_r471/cot_shard1.parquet")):
        Hhat = fit_omr_prior(load_omr_counts(os.path.join(WS, path)))
        cert = build(Hhat)
        tau, gmax_curve, zg_curve, gvec_curve = [], [], [], []
        for a in grid:
            g = [dpflip(K, cert, a)[0] for K in range(33)]
            gvec_curve.append(g)
            tau.append(tau_star(Hhat, g, a))
            gmax_curve.append(max(g))
            zg_curve.append(sum(Hhat[K] for K in range(33) if g[K] <= 1e-12))
        ref = {str(a): round(tau_star(Hhat, [dpflip(K, cert, a)[0]
                                             for K in range(33)], a), 5)
               for a in AGRID_REF}
        curves[cname] = {"tau": tau, "g_max": gmax_curve, "zero_g": zg_curve,
                         "gvec": gvec_curve, "ref": ref}
        print(cname, "ref:", ref)

    # --- RLVE N=8
    mod, Hrl, cert_rl = load_rlve()
    tau, gmax_curve, zg_curve, gvec_curve = [], [], [], []
    for a in grid:
        g = [mod.dp_adaptive_flip(K, cert_rl, a)[0] for K in range(9)]
        gvec_curve.append(g)
        tau.append(tau_star(Hrl, g, a))
        gmax_curve.append(max(g))
        zg_curve.append(sum(Hrl[K] for K in range(9) if g[K] <= 1e-12))
    ref = {str(a): round(tau_star(Hrl, [mod.dp_adaptive_flip(K, cert_rl, a)[0]
                                        for K in range(9)], a), 5)
           for a in AGRID_REF}
    curves["rlve"] = {"tau": tau, "g_max": gmax_curve, "zero_g": zg_curve,
                      "gvec": gvec_curve, "ref": ref}
    print("rlve ref:", ref)

    # --- OpenR1 M=2 (closed form; grid values exact, not approximated)
    Ho = openr1_prior()
    tau, gmax_curve, zg_curve, gvec_curve = [], [], [], []
    for a in grid:
        g = openr1_g(a)
        gvec_curve.append(g)
        tau.append(tau_star(Ho, g, a))
        gmax_curve.append(max(g))
        zg_curve.append(sum(Ho[K] for K in range(3) if g[K] <= 1e-12))
    ref = {str(a): round(tau_star(Ho, openr1_g(a), a), 5) for a in AGRID_REF}
    curves["openr1"] = {"tau": tau, "g_max": gmax_curve, "zero_g": zg_curve,
                        "gvec": gvec_curve, "ref": ref,
                        "certs": [CERT_X0, CERT_X1]}
    print("openr1 ref:", ref)

    # --- P3: reference-level reproduction against r481/r482/r483 JSONs
    r481 = json.load(open(os.path.join(
        WS, "earlystop_drift_r481", "tv_robustness_r481_result.json")))
    r482 = json.load(open(os.path.join(
        WS, "earlystop_drift_r482", "tv_robustness_rlve_r482_result.json")))
    r483 = json.load(open(os.path.join(
        WS, "earlystop_drift_r483", "tv_robustness_openr1_r483_result.json")))
    repro = {
        "omr_shard0": all(abs(curves["omr_shard0"]["ref"][a]
                              - r481["shard0"]["critical_radius"][a]["tau_star"]) < 1e-3
                          for a in ("0.1", "0.05", "0.02", "0.01")),
        "omr_shard1": all(abs(curves["omr_shard1"]["ref"][a]
                              - r481["shard1"]["critical_radius"][a]["tau_star"]) < 1e-3
                          for a in ("0.1", "0.05", "0.02", "0.01")),
        "rlve": all(abs(curves["rlve"]["ref"][a]
                        - r482["critical_radius"][a]["tau_star"]) < 1e-3
                    for a in ("0.1", "0.05", "0.02", "0.01")),
        "openr1": all(abs(curves["openr1"]["ref"][a]
                          - r483["critical_radius"][a]["tau_star_closed"]) < 1e-3
                      for a in ("0.1", "0.05", "0.02", "0.01")),
    }
    out["reference_reproduced"] = repro
    print("reference_reproduced:", repro)

    # --- analytic breakpoints: alpha values where the RULE changes, detected
    # on the full g-vector (self-audit: keying on g_max misses rule changes
    # that leave g_max unchanged -- the RLVE .0075->.008 rule change moves
    # V(0.368) 0.00750->0.00856 at identical g_max=0.01786)
    def breakpoints(name):
        bps = []
        gv = curves[name]["gvec"]
        for i in range(1, len(grid)):
            if any(abs(a - b) > 1e-12 for a, b in zip(gv[i], gv[i - 1])):
                bps.append(round(grid[i], 5))
        return bps

    bps = {name: breakpoints(name) for name in curves}
    out["breakpoints_alpha"] = bps
    out["n_breakpoints"] = {k: len(v) for k, v in bps.items()}
    for k, v in bps.items():
        print(k, "n_breakpoints:", len(v), "first/last:", v[:3], v[-3:])

    # --- P1 verification: tau* non-decreasing WITHIN each constant-rule
    # interval; any strict decrease must sit exactly at a detected breakpoint
    p1 = {}
    for name in curves:
        bp_idx = {grid.index(b) for b in bps[name] if b in grid}
        # grid.index needs exact float match; use rounded lookup instead
        bp_idx = {i for i in range(len(grid)) if round(grid[i], 5) in
                  set(bps[name])}
        viol = [i for i in range(len(grid) - 1)
                if curves[name]["tau"][i + 1] < curves[name]["tau"][i] - 1e-9
                and (i + 1) not in bp_idx]
        p1[name] = {"within_interval_nondecreasing": len(viol) == 0,
                    "violations": [round(grid[i], 4) for i in viol[:5]]}
    out["p1_within_interval_monotone"] = p1
    print("p1:", json.dumps(p1))

    # --- OpenR1 analytic check of P2 jumps
    def openr1_cf(a):
        g = openr1_g(a)[1]
        if g <= 0:
            return 1.0
        base = Ho[1] * g
        if base > a:
            return 0.0
        return min((a - base) / g, Ho[0] + Ho[2])

    eps = 1e-6
    p2 = {
        "jump_down_at_0.07852": openr1_cf(CERT_X0 - eps) - openr1_cf(CERT_X0 + eps),
        "jump_up_at_0.04598": openr1_cf(CERT_X1 + eps) - openr1_cf(CERT_X1 - eps),
        "below": openr1_cf(CERT_X0 - eps), "above": openr1_cf(CERT_X0 + eps),
        "closed_form_matches_grid": all(
            abs(openr1_cf(a) - t) < 1e-3
            for a, t in zip(grid[::40], curves["openr1"]["tau"][::40])),
    }
    out["openr1_jump_analysis"] = p2
    print("openr1 jumps:", {k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in p2.items()})

    # --- P4: descriptive Spearman between zero-g mass and tau* across the
    # 4 carriers x 4 reference levels = 16 cells (shard0/shard1 separate);
    # tie-aware midranks (zero-g mass ties, e.g. RLVE .02==.01, are common)
    def spearman(x, y):
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(v):
                j = i
                while j + 1 < len(v) and abs(v[order[j + 1]] - v[order[i]]) < 1e-12:
                    j += 1
                for k in range(i, j + 1):
                    r[order[k]] = (i + j) / 2.0
                i = j + 1
            return r
        dx, dy = ranks(x), ranks(y)
        n = len(x)
        mx, my = sum(dx) / n, sum(dy) / n
        num = sum((a - mx) * (b - my) for a, b in zip(dx, dy))
        den = (sum((a - mx) ** 2 for a in dx)
               * sum((b - my) ** 2 for b in dy)) ** 0.5
        return num / den if den else None

    zg_cells, tau_cells = [], []
    for name in ("omr_shard0", "omr_shard1", "rlve", "openr1"):
        for a in AGRID_REF:
            i = grid.index(round(a, 6)) if round(a, 6) in grid else min(
                range(len(grid)), key=lambda j: abs(grid[j] - a))
            zg_cells.append(curves[name]["zero_g"][i])
            tau_cells.append(curves[name]["tau"][i])
    out["zerog_tau_spearman_16cells"] = round(spearman(zg_cells, tau_cells), 4)
    print("spearman(zero_g, tau*) over 16 cells:",
          out["zerog_tau_spearman_16cells"])

    # --- curve summary stats for the paper
    summ = {}
    for name in curves:
        t = curves[name]["tau"]
        summ[name] = {"tau_at_0.001": round(t[0], 5), "tau_at_0.20": round(t[-1], 5),
                      "max_tau": round(max(t), 5), "min_tau": round(min(t), 5),
                      "nonmonotone": any(t[i + 1] > t[i] + 1e-9
                                         for i in range(len(t) - 1))}
    out["curve_summary"] = summ
    print("curve_summary:", json.dumps(summ, indent=1))

    # ------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    style = {"omr_shard0": ("C0", "-", "OMR shard 0 ($N{=}32$)"),
             "omr_shard1": ("C0", "--", "OMR shard 1 ($N{=}32$)"),
             "rlve": ("C2", "-", "RLVE ($N{=}8$)"),
             "openr1": ("C3", "-", "OpenR1 ($M{=}2$)")}
    for name, (c, ls, lab) in style.items():
        ax.plot(grid, curves[name]["tau"], color=c, ls=ls, lw=1.8, label=lab)
    # reference levels as light tick marks on the x axis (not full lines)
    for a in AGRID_REF:
        ax.plot([a], [0.012], marker="|", color="0.45", ms=9, mew=1.2,
                zorder=5, clip_on=False)
    ax.annotate("reference levels\n$\\alpha\\in\\{.10,.05,.02,.01\\}$",
                xy=(0.10, 0.012), xytext=(0.128, 0.055), fontsize=7.5,
                color="0.35",
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.6))
    # mark OpenR1 rule-change breakpoints
    for x in (CERT_X0, CERT_X1):
        ax.axvline(x, color="C3", lw=0.9, ls=":", zorder=0)
    ax.annotate("OpenR1 rule changes\n$c_{\\widehat H}=0.0785,\\ 0.0460$",
                xy=(CERT_X0, 0.93), xytext=(0.115, 0.72), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
    ax.set_xlabel(r"certificate level $\alpha$")
    ax.set_ylabel(r"critical radius $\tau^*(\alpha)$")
    ax.set_xlim(0, 0.205)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.set_title("How much prior shift the frozen certificate provably tolerates",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200)
    print("wrote", FIG)

    # curve JSON stays compact: drop dense g_max/zero_g (keep tau + refs)
    slim = {k: {"tau": [round(v, 6) for v in c["tau"]], "ref": c["ref"]}
            for k, c in curves.items()}
    slim["openr1"]["certs"] = [CERT_X0, CERT_X1]
    out["curves"] = slim
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
