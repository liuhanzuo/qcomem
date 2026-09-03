#!/usr/bin/env python3
"""A11 r491: how large must the FIT pool be? tau* as a function of prior-fit
size m, and a theorem-level closed-loop check of Proposition prop:tv along
the estimation axis.

MOTIVATION (same-question follow-up, no new data, zero TEST reads):
prop:tv quantifies tolerable prior shift R for a FROZEN certificate, but the
certificate itself is built from Hhat fitted on m FIT problems (OMR m=4000,
RLVE m=3000). No existing artifact says how tau* degrades when m is smaller
-- i.e., how much FIT data buys how much robustness. This round computes
tau*_m(alpha) exactly at nested FIT subsizes and verifies the proposition's
guarantee across the m axis.

DESIGN (pre-registered BEFORE first run):
  Carriers : OMR shard0 (N=32, FIT 4000), OMR shard1 (N=32, FIT 4000),
             RLVE (N=8, FIT 3000). Same frozen splits as r469/r471/r474:
             Random(20260815).shuffle(idx); FIT = first 4000 (OMR) / 3000
             (RLVE) of the shuffled order.
  Subsizes : m = full * {1/32, 1/16, 1/8, 1/4, 1/2, 1} (floored), as NESTED
             PREFIXES of the frozen shuffled FIT order, so Hhat(m=full) IS
             the frozen paper prior bit-exactly and smaller m are strict
             subsets. This is deterministic: it isolates the m effect from
             resampling noise (disclosed limitation: prefix subsampling is
             one particular nested sequence, not a bootstrap distribution).
  Per (carrier, m, alpha in {.10,.05,.02,.01}):
    - cert_m table from Hhat_m; g_m(K;alpha) by the same exact DP as r481.
    - tau*_m(alpha) = max{R : V_m(R) <= alpha}, exact two-sided greedy LP
      (same worst_case as r484) + 60-iter bisection, same as r484.
    - TV_m = TV(Hhat_m, Hhat_full) = 0.5 * L1.
    - flip_full(rule_m, alpha) = sum_K Hhat_full[K] * g_m(K; alpha): the
      exact flip of the m-fitted rule under the full-data prior.

PREDICTIONS (pre-registered):
  P1 (monotone-up): tau*_m(alpha) is non-decreasing in m (ties allowed) in
      >= 10 of 12 (carrier, alpha) cells; any strict decrease is reported.
  P2 (root-m): TV_m ~ C/sqrt(m): the ratio max/min of TV_m*sqrt(m) across m
      within a carrier is < 3.
  P3 (closed loop, theorem-level): for every (carrier, m, alpha) with
      TV_m <= tau*_m(alpha), flip_full(rule_m, alpha) <= alpha + 1e-12.
      This MUST hold: TV_m <= tau*_m means Hhat_full lies inside the
      tolerance ball of the m-certificate, so prop:tv applies. Predicted
      zero violations; any violation indicates a machinery bug, not a
      theorem counterexample (the proposition is proved in app:proofs).
  P4 (operational): the smallest m with tau*_m(0.05) >= 0.042 (the observed
      OMR cross-shard prior transfer) is <= 2000 on both OMR shards.

Outputs: prior_fit_size_r491_result.json. stdlib + pandas/pyarrow via the
frozen r469/r474 machinery. No GPU/net. Readback: stdout + JSON.
"""
import json, os, random, importlib.util
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
OUT = os.path.join(HERE, "prior_fit_size_r491_result.json")
SEED = 20260815
AGRID = [0.10, 0.05, 0.02, 0.01]
FRACS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
XSHARD_TRANSFER = 0.042  # observed OMR cross-shard prior TV (paper sec:drift)


# ------------------------------------------------------- shared exact LP
def worst_case(H, g, R):
    """Two-sided greedy LP optimum over simplex intersect L1 ball (2R).
    Identical algorithm to r484 (verbatim copy, self-contained)."""
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


def tv(H1, H2):
    return 0.5 * sum(abs(a - b) for a, b in zip(H1, H2))


# ------------------------------------------------------- OMR machinery
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


# ------------------------------------------------------- RLVE machinery
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
    fit_idx = idx[:3000]
    Hfull = mod.fit_prior(Ks, fit_idx)
    return mod, Ks, fit_idx, Hfull


def main():
    build_omr, dpflip_omr = load_r469_machinery()
    out = {"seed": SEED, "fracs": FRACS, "alphas": AGRID,
           "xshard_transfer": XSHARD_TRANSFER,
           "definition": "tau*_m(alpha)=max{R:V_m(R)<=alpha} with cert/g "
                         "rebuilt from the nested-prefix m-fit prior; "
                         "flip_full=exact flip of the m-rule under the "
                         "full-data prior"}

    carriers = {}

    # ---------------- OMR shards
    for cname, path in (("omr_shard0", "earlystop_drift_r467/cot_shard0.parquet"),
                        ("omr_shard1", "earlystop_drift_r471/cot_shard1.parquet")):
        Ks = load_omr_counts(os.path.join(WS, path))
        fit_idx = omr_fit_order(Ks, 4000)
        m_full = len(fit_idx)
        ms = sorted({max(8, int(m_full * f)) for f in FRACS})
        Hfull = hist_prior(Ks, fit_idx, 32)
        cells = []
        for m in ms:
            Hm = hist_prior(Ks, fit_idx[:m], 32)
            if m == m_full:
                assert Hm == Hfull, f"{cname} m=full mismatch"
            cert = build_omr(Hm)
            tvm = tv(Hm, Hfull)
            for a in AGRID:
                g = [dpflip_omr(K, cert, a)[0] for K in range(33)]
                ts = tau_star(Hm, g, a)
                flip_full = sum(Hfull[K] * g[K] for K in range(33))
                cells.append({"m": m, "alpha": a, "tau_star": round(ts, 6),
                              "tv_m": round(tvm, 6),
                              "flip_full": round(flip_full, 8)})
        carriers[cname] = {"ms": ms, "cells": cells}
        print(cname, "ms:", ms)
        for a in AGRID:
            chk = [(r["m"], r["flip_full"]) for r in cells
                   if r["alpha"] == a]
            print(cname, "a=", a, "flip_full:", chk)

    # ---------------- RLVE
    mod, Ks_rl, fit_idx_rl, Hfull_rl = load_rlve()
    m_full = len(fit_idx_rl)
    ms = sorted({max(8, int(m_full * f)) for f in FRACS})
    cells = []
    for m in ms:
        Hm = mod.fit_prior(Ks_rl, fit_idx_rl[:m])
        if m == m_full:
            assert max(abs(a - b) for a, b in zip(Hm, Hfull_rl)) < 1e-15, \
                "rlve m=full must reproduce frozen prior"
        cert = mod.build_cert_table(Hm)
        tvm = tv(Hm, Hfull_rl)
        for a in AGRID:
            g = [mod.dp_adaptive_flip(K, cert, a)[0] for K in range(9)]
            ts = tau_star(Hm, g, a)
            flip_full = sum(Hfull_rl[K] * g[K] for K in range(9))
            cells.append({"m": m, "alpha": a, "tau_star": round(ts, 6),
                          "tv_m": round(tvm, 6),
                          "flip_full": round(flip_full, 8)})
    carriers["rlve"] = {"ms": ms, "cells": cells}
    print("rlve ms:", ms)

    out["carriers"] = carriers

    # ================= prediction checks =================
    checks = {}

    # P1: tau* non-decreasing in m per (carrier, alpha)
    p1 = {}
    for cname, c in carriers.items():
        for a in AGRID:
            seq = [r for r in c["cells"] if r["alpha"] == a]
            seq.sort(key=lambda r: r["m"])
            viol = [(seq[i]["m"], seq[i + 1]["m"])
                    for i in range(len(seq) - 1)
                    if seq[i + 1]["tau_star"] < seq[i]["tau_star"] - 1e-6]
            p1[f"{cname}@{a}"] = {"ok": len(viol) == 0, "viol": viol[:3]}
    n_ok = sum(1 for v in p1.values() if v["ok"])
    checks["p1_monotone_cells"] = {"n_ok": n_ok, "n_total": len(p1),
                                   "pass": n_ok >= 10, "detail": p1}

    # P2: TV_m ~ C/sqrt(m)
    p2 = {}
    for cname, c in carriers.items():
        vals = [(r["m"], r["tv_m"]) for r in c["cells"] if r["alpha"] == AGRID[0]]
        vals = sorted(set((m, t) for m, t in vals))
        scaled = [t * sqrt(m) for m, t in vals if t > 0]
        p2[cname] = {"scaled_min": round(min(scaled), 5),
                     "scaled_max": round(max(scaled), 5),
                     "ratio": round(max(scaled) / min(scaled), 3)}
    checks["p2_rootm"] = {"pass": all(v["ratio"] < 3 for v in p2.values()),
                          "detail": p2}

    # P3: closed loop -- TV_m <= tau*_m(alpha) implies flip_full <= alpha
    p3_viol = []
    p3_n_applicable = 0
    for cname, c in carriers.items():
        for r in c["cells"]:
            if r["tv_m"] <= r["tau_star"] + 1e-9:
                p3_n_applicable += 1
                if r["flip_full"] > r["alpha"] + 1e-12:
                    p3_viol.append({"carrier": cname, **r})
    checks["p3_closed_loop"] = {"n_applicable": p3_n_applicable,
                                "n_violations": len(p3_viol),
                                "pass": len(p3_viol) == 0,
                                "viol": p3_viol[:5]}

    # P4: smallest m with tau*_m(0.05) >= 0.042 on OMR shards
    p4 = {}
    for cname in ("omr_shard0", "omr_shard1"):
        seq = [r for r in carriers[cname]["cells"] if r["alpha"] == 0.05]
        seq.sort(key=lambda r: r["m"])
        mstar = next((r["m"] for r in seq
                      if r["tau_star"] >= XSHARD_TRANSFER), None)
        p4[cname] = {"m_star": mstar}
    checks["p4_operational"] = {
        "pass": all(v["m_star"] is not None and v["m_star"] <= 2000
                    for v in p4.values()),
        "detail": p4}

    out["checks"] = checks
    allpass = all(v["pass"] for v in checks.values())
    out["all_predictions_pass"] = allpass
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "detail"}
                      for k, v in checks.items()}, indent=1))
    print("ALL PASS:", allpass)

    # summary table for the paper: tau* at alpha=.05/.01 across m
    for cname, c in carriers.items():
        for a in (0.05, 0.01):
            row = [(r["m"], r["tau_star"], r["tv_m"], r["flip_full"])
                   for r in c["cells"] if r["alpha"] == a]
            row.sort()
            print(cname, f"alpha={a}:", [(m, t) for m, t, _, _ in row])

    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
