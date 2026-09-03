#!/usr/bin/env python3
"""A11 r494: rule-channel decomposition of the m-axis non-monotonicity.

SAME-QUESTION follow-up to r491 (app:tau (e)): r491 showed tau*_m(alpha) is
NOT monotone in the prior-fit size m and attributed this to two moving
parts: growing m moves the prior (estimation channel, better) AND the
frozen rule (rule channel, larger g_max possible). r491 did NOT separate
the two. This round computes the counterfactual curves that isolate each
channel:
  - fixed-full-prior: tau*_m^{prior-fixed}(alpha) = tau*(Hhat_full, g_m;
    alpha) -- only the rule moves with m, prior frozen at the full fit.
  - fixed-full-rule:  tau*_m^{rule-fixed}(alpha)  = tau*(Hhat_m, g_full;
    alpha) -- only the prior moves, rule frozen at the full fit.
  - both-moving (reference, re-derived): tau*(Hhat_m, g_m; alpha), must
    reproduce the frozen r491 tau*_m grid BIT-EXACTLY (independent
    re-derivation regen anchor).

PREDICTIONS (pre-registered BEFORE first run; mirrored in the checker):
  Q1 (rule channel explains the non-monotonicity): at OMR shard1
     alpha=.01, tau*_m^{prior-fixed} is ALSO non-monotone in m (some
     strict decrease between consecutive tested m), matching the r491
     both-moving shape up-down.
  Q2 (estimation channel is benign): for every tested (carrier, m, alpha),
     tau*_m^{rule-fixed}(alpha) >= tau*_full(alpha) - 1e-6, i.e. holding
     the rule fixed, a COARSER prior never shrinks the certified radius
     below the full-fit radius. Mechanism: with g fixed, tau* shrinks only
     if the base flip sum_K Hhat_m[K] g[K] exceeds the full-prior base;
     the g-profile is decreasing-ish in K (early stop on likely-correct)
     and the coarse prefix over-weights low-K modes relative to the full
     prior, so the coarse base is no larger. PREDICTED: zero violations.
     Any violation is reported as a refinement of the channel story, not
     hidden.
  Q3 (channel dominance at the exemplar): at OMR shard1 alpha=.01 the
     prior-fixed curve reproduces the m=1000 overshoot: tau*_{m=1000}^
     {prior-fixed} > tau*_{m=4000}^{prior-fixed}.

DESIGN: identical machinery to r491 (same frozen FIT order, same nested
prefixes, same exact LP/bisection). OMR shards + RLVE, m grid =
{1/32,...,1} x m_full, alpha in {.10,.05,.02,.01}. No new data, no GPU,
no TEST reads. Output: rule_channel_r494_result.json + stdout readback.
"""
import json, os, random, importlib.util
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
OUT = os.path.join(HERE, "rule_channel_r494_result.json")
SEED = 20260815
AGRID = [0.10, 0.05, 0.02, 0.01]
FRACS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]


def worst_case(H, g, R):
    """Two-sided greedy LP optimum over simplex intersect L1 ball (2R).
    Identical algorithm to r484/r491 (verbatim copy, self-contained)."""
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


def omr_fit_order(Ks, n_fit):
    rnd = random.Random(SEED)
    idx = list(range(len(Ks)))
    rnd.shuffle(idx)
    return idx[:n_fit]


def hist_prior(Ks, idx, n):
    H = [0.0] * (n + 1)
    for i in idx:
        H[Ks[i]] += 1.0
    return [h / len(idx) for h in H]


def load_rlve():
    spec = importlib.util.spec_from_file_location(
        "r474", os.path.join(WS, "earlystop_drift_r474", "rlve_n8_r474.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SHARDS = [os.path.join(WS, "earlystop_drift_r474",
                               f"rlve_p{i}.parquet") for i in range(6)]
    Ks, seqs, qenv = mod.load_data()
    rnd = random.Random(SEED)
    idx = list(range(len(Ks)))
    rnd.shuffle(idx)
    fit_idx = idx[:3000]
    Hfull = mod.fit_prior(Ks, fit_idx)
    return mod, Ks, fit_idx, Hfull


def decompose(cname, priors, certs, dpflip, n_sym):
    """priors/certs: dict m -> H_m / cert_m (m=m_full key must exist).
    Returns cells with the three tau* variants per (m, alpha)."""
    m_full = max(priors)
    Hfull = priors[m_full]
    gfull = {a: [dpflip(K, certs[m_full], a)[0] for K in range(n_sym + 1)]
             for a in AGRID}
    cells = []
    for m in sorted(priors):
        Hm = priors[m]
        gm = {a: [dpflip(K, certs[m], a)[0] for K in range(n_sym + 1)]
              for a in AGRID}
        for a in AGRID:
            both = tau_star(Hm, gm[a], a)
            prior_fixed = tau_star(Hfull, gm[a], a)
            rule_fixed = tau_star(Hm, gfull[a], a)
            base_m_rulefull = sum(Hfull[K] * gm[a][K]
                                  for K in range(n_sym + 1))
            base_full_rulefull = sum(Hfull[K] * gfull[a][K]
                                     for K in range(n_sym + 1))
            cells.append({
                "m": m, "alpha": a,
                "tau_both": round(both, 6),
                "tau_prior_fixed": round(prior_fixed, 6),
                "tau_rule_fixed": round(rule_fixed, 6),
                "base_flip_rule_m": round(base_m_rulefull, 8),
                "base_flip_rule_full": round(base_full_rulefull, 8),
                "gmax_m": round(max(gm[a]), 8),
                "gmax_full": round(max(gfull[a]), 8),
            })
    return cells, m_full


def main():
    build_omr, dpflip_omr = load_r469_machinery()
    out = {"seed": SEED, "fracs": FRACS, "alphas": AGRID,
           "definition": "tau_prior_fixed = tau*(Hhat_full, g_m); "
                         "tau_rule_fixed = tau*(Hhat_m, g_full); "
                         "tau_both = tau*(Hhat_m, g_m) (regen anchor vs r491)"}
    carriers = {}

    for cname, path in (("omr_shard0", "earlystop_drift_r467/cot_shard0.parquet"),
                        ("omr_shard1", "earlystop_drift_r471/cot_shard1.parquet")):
        Ks = load_omr_counts(os.path.join(WS, path))
        fit_idx = omr_fit_order(Ks, 4000)
        m_full = len(fit_idx)
        ms = sorted({max(8, int(m_full * f)) for f in FRACS})
        priors = {m: hist_prior(Ks, fit_idx[:m], 32) for m in ms}
        certs = {m: build_omr(priors[m]) for m in ms}
        cells, _ = decompose(cname, priors, certs, dpflip_omr, 32)
        carriers[cname] = {"ms": ms, "cells": cells}

    mod, Ks_rl, fit_idx_rl, Hfull_rl = load_rlve()
    m_full_rl = len(fit_idx_rl)
    ms_rl = sorted({max(8, int(m_full_rl * f)) for f in FRACS})
    priors_rl = {m: mod.fit_prior(Ks_rl, fit_idx_rl[:m]) for m in ms_rl}
    certs_rl = {m: mod.build_cert_table(priors_rl[m]) for m in ms_rl}
    cells_rl, _ = decompose("rlve", priors_rl, certs_rl,
                            mod.dp_adaptive_flip, 8)
    carriers["rlve"] = {"ms": ms_rl, "cells": cells_rl}

    out["carriers"] = carriers

    # ---------------- regen anchor vs r491 frozen artifact ----------------
    r491 = json.load(open(os.path.join(
        WS, "earlystop_drift_r491", "prior_fit_size_r491_result.json")))
    anchor_mismatch = []
    for cn, c in carriers.items():
        old = {(r["m"], r["alpha"]): r["tau_star"]
               for r in r491["carriers"][cn]["cells"]}
        for cell in c["cells"]:
            key = (cell["m"], cell["alpha"])
            if abs(old[key] - cell["tau_both"]) > 1e-6:
                anchor_mismatch.append({"carrier": cn, "m": key[0],
                                        "alpha": key[1],
                                        "r491": old[key],
                                        "r494": cell["tau_both"]})
    out["regen_anchor_r491"] = {"n_mismatch": len(anchor_mismatch),
                                "mismatch": anchor_mismatch[:5]}

    # ---------------- prediction checks ----------------
    def curve(cn, a, field):
        seq = [r for r in carriers[cn]["cells"] if r["alpha"] == a]
        seq.sort(key=lambda r: r["m"])
        return [(r["m"], r[field]) for r in seq]

    checks = {}

    # Q1: prior-fixed curve at OMR s1 @.01 is also non-monotone
    pf = curve("omr_shard1", 0.01, "tau_prior_fixed")
    dec = [(pf[i][0], pf[i + 1][0]) for i in range(len(pf) - 1)
           if pf[i + 1][1] < pf[i][1] - 1e-6]
    checks["q1_prior_fixed_nonmono_s1_01"] = {
        "pass": len(dec) > 0, "curve": pf, "decreases": dec}

    # Q2: rule-fixed never below full-fit radius (all carriers/m/alpha)
    q2_viol = []
    for cn, c in carriers.items():
        for cell in c["cells"]:
            full = next(r for r in c["cells"]
                        if r["alpha"] == cell["alpha"]
                        and r["m"] == max(x["m"] for x in c["cells"]))
            if cell["tau_rule_fixed"] < full["tau_rule_fixed"] - 1e-6:
                q2_viol.append({"carrier": cn, "m": cell["m"],
                                "alpha": cell["alpha"],
                                "rule_fixed": cell["tau_rule_fixed"],
                                "full": full["tau_rule_fixed"]})
    checks["q2_rule_fixed_never_below_full"] = {
        "pass": len(q2_viol) == 0, "n_viol": len(q2_viol),
        "viol": q2_viol[:8]}

    # Q3: prior-fixed reproduces the m=1000 overshoot at s1 @.01
    by = {(r["m"], r["alpha"]): r
          for r in carriers["omr_shard1"]["cells"]}
    checks["q3_overshoot_prior_fixed"] = {
        "pass": by[(1000, 0.01)]["tau_prior_fixed"]
                > by[(4000, 0.01)]["tau_prior_fixed"],
        "m1000": by[(1000, 0.01)]["tau_prior_fixed"],
        "m4000": by[(4000, 0.01)]["tau_prior_fixed"]}

    out["checks"] = checks
    print(json.dumps({k: v for k, v in checks.items()}, indent=1))
    print("regen anchor mismatches:", len(anchor_mismatch))

    # paper-facing summary rows
    for cn in carriers:
        for a in (0.05, 0.01):
            rows = [(r["m"], r["tau_both"], r["tau_prior_fixed"],
                     r["tau_rule_fixed"], r["base_flip_rule_m"],
                     r["gmax_m"])
                    for r in carriers[cn]["cells"] if r["alpha"] == a]
            rows.sort()
            print(cn, f"a={a}: (m, both, prior_fixed, rule_fixed, base, gmax)")
            for r in rows:
                print("   ", r)

    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
