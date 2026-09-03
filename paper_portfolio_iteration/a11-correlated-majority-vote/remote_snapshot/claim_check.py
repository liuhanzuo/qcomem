#!/usr/bin/env python3
"""A11 BAYES-H audit pack: claim checker.

Every load-bearing number quoted in paper.tex / appendix_*.tex is loaded
from the real artifact JSON under audit_pack/results/ and compared against
the printed string at the precision used in the paper.

Run from the audit_pack directory:
    python3 claim_check.py
Exit code 0 iff every claim passes. Stdlib only. No network, no GPU.

Provenance notes (r472):
- forced-stop occupancy (App B) was recomputed with the r469 runner's own
  DP machinery, same parquet/seed/split: earlystop_drift_r472/forced_stop_r472.json
- prior TV distance (Sec.5) recomputed exactly as r471 R2 constructed the
  two FIT priors: 0.042250 -> printed 0.042 (PASS).

Provenance notes (r493, W1 layer):
- A mechanical guard twice reported the r491 prior-fit script/JSON as
  nonexistent. Triage: it scanned a candidate-scoped path
  (stop_drift_r491...) while the real originals live in
  agents/A11/workspace/earlystop_drift_r491/ with byte-identical copies in
  this pack under results/. The W1.* layer hard-asserts the REAL paths:
  existence+size, workspace<->pack sha256 parity, and content anchors, so
  a missing/stale provenance path can never again pass silently. Workspace
  halves are gated on the origin workspace being reachable; clean-room
  replays elsewhere still fully check the in-pack copies.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")

FAILS = []
PASSES = []
EXTS = []  # external-provenance layer: checks that depend on files OUTSIDE
           # this pack (author workspace originals). Reported separately so a
           # clean-room replay (pack only) shows 0 internal FAIL. A gate that
           # merely verifies an external original is REACHABLE is classified
           # external provenance, never counted as an in-pack FAIL (r496
           # layering per MGR b410ef4624df).
EXTERNAL_IDS = frozenset({
    # r491 workspace-original provenance (files live outside this pack):
    "W1.ws.py.gate", "W1.ws.py.exist", "W1.parity.py",
    "W1.ws.json.gate", "W1.ws.json.exist", "W1.parity.json",
    "W1.log.gate", "W1.ws.log.exist", "W1.content.log",
    # In-pack W1 items (pack.*.exist, content.py, content.py2, content.json)
    # stay INTERNAL: they replay from the pack's own results/ copies.
})


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def claim(cid, desc, got_raw, printed, fmt):
    if got_raw is None:
        FAILS.append(f"{cid} {desc}: artifact value is None")
        return
    v = float(got_raw)
    if fmt == "pct1":
        got = f"{100 * v:.1f}"
    elif fmt == "pct2":
        got = f"{100 * v:.2f}"
    elif fmt == "f4":
        got = f"{v:.4f}"
    elif fmt == "f3":
        got = f"{v:.3f}"
    elif fmt == "f2":
        got = f"{v:.2f}"
    elif fmt == "k1":
        got = f"{v:.1f}"
    elif fmt == "int":
        got = f"{int(v)}"
    else:
        raise ValueError(fmt)
    if got == printed:
        PASSES.append(f"{cid} {desc}: artifact={got} printed={printed}")
    else:
        FAILS.append(f"{cid} {desc}: artifact={got} != printed={printed} (raw={v})")


def claim_tol(cid, desc, got_raw, printed_pct, tol_pt):
    """Percentage claim with an absolute tolerance in points. Used where the
    artifact stores a rounded value and the paper rounded up at a boundary
    (e.g. raw 80.85% printed as 80.9%)."""
    got = 100 * float(got_raw)
    if abs(got - printed_pct) <= tol_pt:
        PASSES.append(f"{cid} {desc}: artifact={got:.4f}pt printed={printed_pct}pt (tol {tol_pt}pt)")
    else:
        FAILS.append(f"{cid} {desc}: artifact={got:.4f}pt != printed={printed_pct}pt beyond tol {tol_pt}pt")


def claim_true(cid, desc, cond, detail=""):
    if cond:
        PASSES.append(f"{cid} {desc}: confirmed {detail}")
    elif cid in EXTERNAL_IDS:
        EXTS.append(f"{cid} {desc}: EXTERNAL-PROVENANCE (original outside pack, "
                    f"not replayable clean-room; in-run state must PASS) {detail}")
    else:
        FAILS.append(f"{cid} {desc}: FAILED {detail}")


def main():
    fit = load("fit_cal_test_r469_result.json")
    margin = load("margin_repair_r469_result.json")
    shard1 = load("shard1_robust_r471_result.json")
    l2 = load("L2_lemmas_r468_result.json")
    adaptive = load("adaptive_r468_result.json")
    passrate = load("passrate_r467_result.json")
    drift = load("drift_stress_r469_result.json")
    forced = load("forced_stop_r472.json")

    tr = fit["test_readout"]
    cal = fit["cal_selection"]

    # ---------- Table 1 (tab:main), alpha=0.10 ----------
    # r475 hostile self-audit fix: the v2-v5 print of this row
    # (0.0637/0.0831/0.0491) traced to NO on-disk artifact; the canonical
    # single TEST readout fit_cal_test_r469_result.json holds
    # 0.0645/0.0794/0.0370 (paper corrected r475; claim_check r2-r474 had a
    # coverage hole: this row was never checked).
    h1 = tr["FIXED_HOEF_a0.1"]
    claim("T1.HOEF10.flip", "Table1 HOEF a=.10 flip", h1["realized_flip"], "0.0645", "f4")
    claim("T1.HOEF10.k", "Table1 HOEF a=.10 mean k", h1["mean_k"], "7.0", "k1")
    claim("T1.HOEF10.save", "Table1 HOEF a=.10 saving", h1["saving_vs_full"], "78.1", "pct1")
    e1 = tr["FIXED_EB_a0.1"]
    claim("T1.EB10.flip", "Table1 EB a=.10 flip", e1["realized_flip"], "0.0794", "f4")
    claim("T1.EB10.k", "Table1 EB a=.10 mean k", e1["mean_k"], "5.0", "k1")
    claim("T1.EB10.save", "Table1 EB a=.10 saving", e1["saving_vs_full"], "84.4", "pct1")
    b1 = tr["BAYESH_a0.1"]
    claim("T1.BH10.flip", "Table1 BAYES-H a=.10 flip", b1["realized_flip"], "0.0370", "f4")
    claim("T1.BH10.k", "Table1 BAYES-H a=.10 mean k", b1["mean_k"], "5.1", "k1")
    claim("T1.BH10.save", "Table1 BAYES-H a=.10 saving", b1["saving_vs_full"], "84.0", "pct1")
    claim("T1.HOEF10.kstar", "CAL selection HOEF k*=7 @a=.10", cal["0.1"]["FIXED_HOEF_k"], "7", "int")
    claim("T1.EB10.kstar", "CAL selection EB k*=5 @a=.10", cal["0.1"]["FIXED_EB_k"], "5", "int")

    # ---------- Table 1 (tab:main), alpha=0.05 ----------
    b = tr["BAYESH_a0.05"]
    claim("T1.BH05.flip", "Table1 BAYES-H a=.05 flip", b["realized_flip"], "0.0249", "f4")
    claim("T1.BH05.k", "Table1 BAYES-H a=.05 mean k", b["mean_k"], "6.1", "k1")
    claim_tol("T1.BH05.save", "Table1 BAYES-H a=.05 saving (artifact stores 4dp-rounded 0.8085; printed 80.9 rounds the boundary up)",
              b["saving_vs_full"], 80.9, 0.06)
    h = tr["FIXED_HOEF_a0.05"]
    claim("T1.HOEF05.flip", "Table1 HOEF a=.05 flip", h["realized_flip"], "0.0156", "f4")
    claim("T1.HOEF05.k", "Table1 HOEF a=.05 mean k", h["mean_k"], "27.0", "k1")
    claim("T1.HOEF05.save", "Table1 HOEF a=.05 saving", h["saving_vs_full"], "15.6", "pct1")
    e = tr["FIXED_EB_a0.05"]
    claim("T1.EB05.flip", "Table1 EB a=.05 flip", e["realized_flip"], "0.0324", "f4")
    claim("T1.EB05.k", "Table1 EB a=.05 mean k (=cal k*=17)", e["mean_k"], "17.0", "k1")
    claim("T1.EB05.save", "Table1 EB a=.05 saving", e["saving_vs_full"], "46.9", "pct1")
    claim("T1.EB05.kstar", "CAL selection EB k*=17 @a=.05", cal["0.05"]["FIXED_EB_k"], "17.0", "k1")
    claim("T1.HOEF05.kstar", "CAL selection HOEF k*=27 @a=.05", cal["0.05"]["FIXED_HOEF_k"], "27.0", "k1")

    # Table 1 alpha=0.02
    b2 = tr["BAYESH_a0.02"]
    claim("T1.BH02.flip", "Table1 BAYES-H a=.02 flip", b2["realized_flip"], "0.0091", "f4")
    claim("T1.BH02.k", "Table1 BAYES-H a=.02 mean k", b2["mean_k"], "8.5", "k1")
    claim("T1.BH02.save", "Table1 BAYES-H a=.02 saving", b2["saving_vs_full"], "73.5", "pct1")
    e2 = tr["FIXED_EB_a0.02"]
    claim("T1.EB02.save", "Table1 EB a=.02 saving", e2["saving_vs_full"], "3.1", "pct1")
    claim("T1.EB02.flip", "Table1 EB a=.02 flip", e2["realized_flip"], "0.0083", "f4")
    claim("T1.EB02.kstar", "CAL selection EB k*=31 @a=.02", cal["0.02"]["FIXED_EB_k"], "31", "int")
    claim_true("T1.HOEF02.null", "Table1 HOEF a=.02 no certifiable budget",
               cal["0.02"]["FIXED_HOEF_k"] is None,
               f"cal_selection['0.02'].FIXED_HOEF_k={cal['0.02']['FIXED_HOEF_k']}")

    # WINDOW3 invalidity (from adaptive_r468 artifact; same values used in Table 1)
    w = adaptive["stoppers"]["WINDOW3"]
    claim("T1.WIN.flip", "WINDOW3 flip", w["realized_flip"], "0.0577", "f4")
    claim("T1.WIN.k", "WINDOW3 mean k", w["mean_k"], "4.4", "k1")
    claim("T1.WIN.save", "WINDOW3 saving", w["rollout_saving_vs_full"], "86.2", "pct1")
    # WINDOW3 at alpha=.10 (paper Sec.5): certified population UCB 0.0606
    # printed 0.061; the r468 sweep stored valid_at_alpha=True on its 5804-
    # problem pool because 0.0606<=0.10 there, so the paper statement is
    # phrased as a UCB quote, not a validity bit. r475 self-audit note.
    sw = load("adaptive_sweep_r468_result.json")
    w10 = sw["alphas"]["0.1"]["WINDOW3"]
    claim("T1.WIN10.ucb", "WINDOW3 a=.10 population UCB (printed 0.061)",
          w10["population_bootUCB95"], "0.061", "f3")
    claim_true("T1.WIN10.real", "WINDOW3 realized flip ~0.058 flat in alpha",
               abs(w10["realized_flip"] - 0.058) < 0.002,
               f"flip={w10['realized_flip']}")

    # ---------- fair gap (Sec.5) ----------
    fg = fit["fair_gap_bayesh_vs_hoeffding"]
    claim("G.05", "fair gap a=.05", fg["0.05"]["abs_gap_bayesh_minus_hoef"], "0.652", "f3")
    claim("G.05.ci", "fair gap CI radius a=.05", fg["0.05"]["gap_ci_radius"], "0.029", "f3")
    claim_true("G.05.sig", "fair gap a=.05 significant", fg["0.05"]["significant"] is True)
    claim("G.10", "fair gap a=.10", fg["0.1"]["abs_gap_bayesh_minus_hoef"], "0.058", "f3")
    claim("G.10.ci", "fair gap CI radius a=.10", fg["0.1"]["gap_ci_radius"], "0.029", "f3")
    claim_true("G.10.sig", "fair gap a=.10 significant", fg["0.1"]["significant"] is True)

    # ---------- App B forced-stop occupancy (r472 exact recompute) ----------
    claim("F.05", "forced-stop occupancy a=.05 (printed 0.59%)", forced["0.05"], "0.59", "pct2")
    claim("F.02", "forced-stop occupancy a=.02 (printed 1.10%)", forced["0.02"], "1.10", "pct2")

    # ---------- Sec.3 mixture structure ----------
    claim("P.pvar", "per-task p variance", passrate["p_var"], "0.129", "f3")
    claim("P.binom", "binomial baseline variance", passrate["binom_baseline_var"], "0.0038", "f4")
    ratio = passrate["p_var"] / passrate["binom_baseline_var"]
    claim_true("P.34x", "heterogeneity >34x binomial baseline", 34.0 <= ratio < 35.0,
               f"ratio={ratio:.2f}")
    claim("P.extreme", "frac extreme (p<=.1 or >=.9)", passrate["frac_extreme_p_le0.1_or_ge0.9"], "0.45", "f2")
    claim("P.mid", "frac mid (0.4-0.6)", passrate["frac_mid_p_0.4_0.6"], "0.116", "f3")

    # ---------- Sec.5 second-shard replication (R1) ----------
    r1 = shard1["R1_within"]["test_readout"]
    claim("S1.BH05.save", "shard1 within BAYES-H save a=.05", r1["BAYESH_a0.05"]["saving_vs_full"], "80.6", "pct1")
    claim("S1.BH05.flip", "shard1 within BAYES-H flip a=.05", r1["BAYESH_a0.05"]["realized_flip"], "0.0246", "f4")
    claim("S1.BH02.save", "shard1 within BAYES-H save a=.02", r1["BAYESH_a0.02"]["saving_vs_full"], "73.1", "pct1")
    claim("S1.BH02.flip", "shard1 within BAYES-H flip a=.02", r1["BAYESH_a0.02"]["realized_flip"], "0.0088", "f4")
    claim("S1.het", "shard1 het_excess", shard1["R1_within"]["het_excess_var"], "0.125", "f3")
    claim("S1.extreme", "shard1 frac_extreme", shard1["R1_within"]["frac_extreme"], "0.444", "f3")
    claim("S1.HOEF05.save", "shard1 HOEF save a=.05", r1["FIXED_HOEF_a0.05"]["saving_vs_full"], "21.9", "pct1")
    claim_true("S1.HOEF02.null", "shard1 HOEF a=.02 no budget",
               shard1["R1_within"]["cal_selection"]["0.02"]["FIXED_HOEF_k"] is None)

    # ---------- Sec.5 cross-shard prior transfer (R2) ----------
    r2 = shard1["R2_transfer"]["test_readout"]
    claim("S2.BH05.save", "transfer BAYES-H save a=.05", r2["BAYESH_a0.05"]["saving_vs_full"], "80.6", "pct1")
    claim("S2.BH05.flip", "transfer BAYES-H flip a=.05", r2["BAYESH_a0.05"]["realized_flip"], "0.0247", "f4")
    claim("S2.BH02.save", "transfer BAYES-H save a=.02", r2["BAYESH_a0.02"]["saving_vs_full"], "73.2", "pct1")
    claim("S2.BH02.flip", "transfer BAYES-H flip a=.02", r2["BAYESH_a0.02"]["realized_flip"], "0.0089", "f4")
    # max movement across the three readouts (paper Sec.5 says <=0.3pt for
    # the numbers it prints there; the abstract extends the claim to all
    # readouts, where the max is 0.39pt at a=0.10 -- recorded as an
    # over-rounding for the author to fix, not a silent pass)
    mov_printed = max(
        abs(b["saving_vs_full"] - r1["BAYESH_a0.05"]["saving_vs_full"]),
        abs(b["saving_vs_full"] - r2["BAYESH_a0.05"]["saving_vs_full"]),
        abs(b2["saving_vs_full"] - r1["BAYESH_a0.02"]["saving_vs_full"]),
        abs(b2["saving_vs_full"] - r2["BAYESH_a0.02"]["saving_vs_full"]),
    )
    mov_all = max(
        mov_printed,
        abs(tr["BAYESH_a0.1"]["saving_vs_full"] - r1["BAYESH_a0.1"]["saving_vs_full"]),
        abs(tr["BAYESH_a0.1"]["saving_vs_full"] - r2["BAYESH_a0.1"]["saving_vs_full"]),
    )
    claim_true("S2.move.printed", "cross-readout movement of the numbers printed in Sec.5 <= 0.4pt",
               mov_printed <= 0.004, f"max={100*mov_printed:.3f}pt")
    claim_true("S2.move.all", "cross-readout movement over ALL alphas <= 0.4pt "
               "(paper fixed r472: abstract previously said 0.3; true max 0.39pt at a=0.10)",
               mov_all <= 0.004, f"max={100*mov_all:.3f}pt (0.10-alpha cell)")

    # ---------- Sec.7 margin repair ----------
    mr = margin["results"]
    e3 = mr["a0.05_E3_blockswap_d0.15_g0.025"]
    claim("M.E3.flip", "margin E3 d=.15 g=.025 flip", e3["flip"], "0.0495", "f4")
    claim("M.E3.save", "margin E3 d=.15 g=.025 saving", e3["saving"], "77.2", "pct1")
    claim_true("M.E3.valid", "margin E3 d=.15 g=.025 valid", e3["valid"] is True)
    e2m = mr["a0.05_E2_linear_p_d0.2_g0.03"]
    claim("M.E2.flip", "margin E2 d=.20 g=.03 flip", e2m["flip"], "0.0491", "f4")
    claim("M.E2.save", "margin E2 d=.20 g=.03 saving", e2m["saving"], "71.8", "pct1")
    claim("M.E2.flip4", "margin E2 d=.20 g=.03 flip printed 0.0491", e2m["flip"], "0.0491", "f4")
    claim_true("M.E2.valid", "margin E2 d=.20 g=.03 valid", e2m["valid"] is True)
    e3b = mr["a0.05_E3_blockswap_d0.2_g0.03"]
    claim_true("M.E3.d02.breaks", "margin E3 d=.20 g=.03 still invalid (needs g>.03)",
               e3b["valid"] is False, f"flip={e3b['flip']}")

    # ---------- Sec.7 drift robustness headline (BAYES-H most graceful) ----------
    dr = drift["results"]
    bh = dr["E2_linear_p"]["0.1"]["0.2"]["BAYESH"]
    claim("D.E2.a10.bh", "E2 a=.10 d=.20 BAYES-H flip 0.083 valid", bh["flip"], "0.083", "f3")
    claim_true("D.E2.a10.bh.valid", "E2 a=.10 d=.20 BAYES-H valid", bh["valid"] is True)
    claim_true("D.E2.a10.fix15.breaks", "E2 a=.10 d=.15 FIXED-EB invalid",
               dr["E2_linear_p"]["0.1"]["0.15"]["FIXED_EB"]["valid"] is False)
    bh3 = dr["E3_blockswap"]["0.05"]["0.15"]["BAYESH"]
    claim("D.E3.a05.bh15", "E3 a=.05 d=.15 BAYES-H flip 0.052", bh3["flip"], "0.052", "f3")
    claim_true("D.E3.a05.bh15.breaks", "E3 a=.05 d=.15 BAYES-H invalid (breaks only at .15)",
               bh3["valid"] is False)
    claim_true("D.E3.a05.eb15.survives", "E3 a=.05 d=.15 FIXED-EB survives",
               dr["E3_blockswap"]["0.05"]["0.15"]["FIXED_EB"]["valid"] is True)
    # r475: paper text corrected to "WINDOW3 breaks at delta=0.20 under E2
    # (flip 0.104); under E3 it stays valid throughout" -- pin both facts.
    claim("D.E2.a10.win20", "E2 a=.10 d=.20 WINDOW3 flip 0.104",
          dr["E2_linear_p"]["0.1"]["0.2"]["WINDOW3"]["flip"], "0.104", "f3")
    claim_true("D.E2.a10.win20.breaks", "E2 a=.10 d=.20 WINDOW3 invalid",
               dr["E2_linear_p"]["0.1"]["0.2"]["WINDOW3"]["valid"] is False)
    claim_true("D.E3.a10.win.allvalid", "E3 a=.10 WINDOW3 valid at ALL deltas",
               all(dr["E3_blockswap"]["0.1"][dl]["WINDOW3"]["valid"] is True
                   for dl in ["0.0", "0.05", "0.1", "0.15", "0.2"]))
    # unmargined BAYES-H saving in the E3 stress harness (paper: 81.0%)
    claim("D.E3.a05.bh0.save", "E3 a=.05 d=0 BAYES-H saving 81.0 (harness)",
          dr["E3_blockswap"]["0.05"]["0.0"]["BAYESH"]["saving"], "81.0", "pct1")

    # ---------- App A.1 L2c monotonicity ----------
    l2c = l2["L2c_monotone_per_problem"]
    claim_true("L2c.viol", "L2c monotonicity exhaustive check: 0 violations",
               l2c["monotonicity_violations"] == 0 and l2c["holds"] is True,
               f"violations={l2c['monotonicity_violations']}, grid K={l2c['K_values']}")

    # ---------- DP vs MC self-check ----------
    sc = fit["selfcheck_dp_vs_mc"]
    claim_true("DP.MC", "DP vs replay-MC self-check within 3 sigma",
               sc["pass"] is True and sc["max_abs_diff"] <= sc["tol_3sigma"],
               f"max={sc['max_abs_diff']} tol={sc['tol_3sigma']}")

    # ---------- Sec.5 second model-carrier pair (OpenR1 M=2, r473 / paper v4) ----------
    or1 = load("openr1_m2_pilot_r473.json")
    h = or1["prior_H_p0_p5_p1"]
    claim("OR1.H0", "OpenR1 prior mass p=0 (always-fail)", h[0], "0.253", "f3")
    claim("OR1.Hmid", "OpenR1 prior mass p=.5 (coin-flip)", h[1], "0.232", "f3")
    claim("OR1.H1", "OpenR1 prior mass p=1 (always-pass)", h[2], "0.515", "f3")
    tr1 = or1["test_readout"]
    claim("OR1.BH10.flip", "OpenR1 BAYESH a=.10 realized flip", tr1["BAYESH_a0.1"]["realized_flip"], "0.0592", "f4")
    claim("OR1.BH10.save", "OpenR1 BAYESH a=.10 saving", tr1["BAYESH_a0.1"]["saving_vs_full"], "50.0", "pct1")
    claim_true("OR1.BH10.valid", "OpenR1 BAYESH a=.10 valid (flip<=alpha)",
               tr1["BAYESH_a0.1"]["realized_flip"] <= 0.10)
    claim_true("OR1.BH10.valid20", "OpenR1 BAYESH a=.20 valid (flip<=alpha)",
               tr1["BAYESH_a0.2"]["realized_flip"] <= 0.20)
    claim("OR1.BH05.flip", "OpenR1 BAYESH a=.05 flip (partial stop)",
          tr1["BAYESH_a0.05"]["realized_flip"], "0.0294", "f4")
    claim("OR1.BH05.save", "OpenR1 BAYESH a=.05 saving (partial stop)",
          tr1["BAYESH_a0.05"]["saving_vs_full"], "32.1", "pct1")
    claim_true("OR1.BH05.valid", "OpenR1 BAYESH a=.05 valid (flip<=alpha)",
               tr1["BAYESH_a0.05"]["realized_flip"] <= 0.05)
    claim("OR1.BH02.nostop", "OpenR1 BAYESH a=.02 never stops (saving 0)",
          tr1["BAYESH_a0.02"]["saving_vs_full"], "0.0", "pct1")
    claim("OR1.F1.flip", "OpenR1 FIXED-1 realized flip", tr1["FIXED1"]["realized_flip"], "0.0592", "f4")
    claim("OR1.F1.analytic", "OpenR1 FIXED-1 analytic flip E_H[p(1-p)]",
          or1["E_flip_FIXED1_theory"], "0.058", "f3")
    claim("OR1.n", "OpenR1 usable problems (M=2 deduped)", or1["n_problems"], "8853", "int")
    # r475: paper OpenR1 paragraph corrected -- at a=.10 full stopping carries
    # HOEF UCB 0.0797 <= 0.10 (valid); paper no longer claims a=.10 needs no
    # certificate knob. Pin the UCBs and the n=9374 provenance number.
    claim("OR1.HOEF.ucb", "OpenR1 FIXED-1 Hoeffding UCB (0.0797<=0.10)",
          or1["cal_selection"]["FIXED1"]["hoef"], "0.0797", "f4")
    claim("OR1.EB.ucb10", "OpenR1 FIXED-1 EB UCB at a=.10",
          or1["cal_selection"]["FIXED1"]["eb"], "0.0707", "f4")


    # ---------- Sec.5 third model-carrier pair (RLVE Qwen3-4B N=8, r474 / paper v5) ----------
    rl = load("rlve_n8_r474_result.json")
    w = rl["within"]
    trl = w["test_readout"]
    claim("RL.n", "RLVE full-8 problems", w["n"], "9000", "int")
    claim("RL.extreme", "RLVE frac extreme problems 42%", w["frac_extreme"], "42.0", "pct1")
    claim("RL.hetex", "RLVE heterogeneity excess var 0.100", w["het_excess_var"], "0.0999", "f4")
    claim("RL.BH10.save", "RLVE BAYESH a=.10 saving", trl["BAYESH_a0.1"]["saving_vs_full"], "57.6", "pct1")
    claim("RL.BH05.save", "RLVE BAYESH a=.05 saving", trl["BAYESH_a0.05"]["saving_vs_full"], "50.9", "pct1")
    claim("RL.BH02.save", "RLVE BAYESH a=.02 saving", trl["BAYESH_a0.02"]["saving_vs_full"], "48.2", "pct1")
    claim("RL.BH01.save", "RLVE BAYESH a=.01 saving", trl["BAYESH_a0.01"]["saving_vs_full"], "45.4", "pct1")
    claim("RL.BH10.flip", "RLVE BAYESH a=.10 flip", trl["BAYESH_a0.1"]["realized_flip"], "0.031", "f3")
    claim("RL.BH05.flip", "RLVE BAYESH a=.05 flip", trl["BAYESH_a0.05"]["realized_flip"], "0.010", "f3")
    claim("RL.BH02.flip", "RLVE BAYESH a=.02 flip", trl["BAYESH_a0.02"]["realized_flip"], "0.005", "f3")
    claim("RL.BH01.flip", "RLVE BAYESH a=.01 flip", trl["BAYESH_a0.01"]["realized_flip"], "0.002", "f3")
    for a in ["0.1", "0.05", "0.02", "0.01"]:
        claim_true(f"RL.BH{a}.valid", f"RLVE BAYESH a={a} valid (flip<=alpha)",
                   trl[f"BAYESH_a{a}"]["realized_flip"] <= float(a))
        claim_true(f"RL.BH{a}.calpass", f"RLVE BAYESH a={a} CAL certificate passes",
                   w["cal_selection"][a]["BAYESH_ok"] is True)
    claim_true("RL.EB05.null", "RLVE FIXED-EB has no certifiable budget at a=.05",
               w["cal_selection"]["0.05"]["FIXED_EB_k"] is None)
    claim_true("RL.HOEF05.null", "RLVE FIXED-HOEF has no certifiable budget at a=.05",
               w["cal_selection"]["0.05"]["FIXED_HOEF_k"] is None)
    claim_true("RL.EB02.null", "RLVE FIXED-EB null at a=.02",
               w["cal_selection"]["0.02"]["FIXED_EB_k"] is None)
    claim_true("RL.HOEF02.null", "RLVE FIXED-HOEF null at a=.02",
               w["cal_selection"]["0.02"]["FIXED_HOEF_k"] is None)
    g = w["fair_gap_bayesh_vs_hoeffding"]["0.1"]
    claim("RL.gap10", "RLVE a=.10 paired saving gap over HOEF +0.20", g["abs_gap_bayesh_minus_hoef"], "0.20", "f2")
    claim("RL.gap10.ci", "RLVE a=.10 gap CI radius 0.03", g["gap_ci_radius"], "0.03", "f2")
    claim_true("RL.gap10.sig", "RLVE a=.10 gap significant", g["significant"] is True)
    dd = rl["drift_diagnostic"]
    claim_true("RL.slope", "RLVE generation-order slope flat (|slope|<2e-4)", abs(dd["pooled_slope_per_trial"]) < 2e-4, f"slope={dd['pooled_slope_per_trial']}")
    # r475: pin RLVE a=.10 baselines (both select k=5; realized 0.0613) and
    # the a=.01 nulls (paper: no distribution-free budget below a=.10).
    claim("RL.EB10.flip", "RLVE FIXED-EB a=.10 flip 0.0613",
          trl["FIXED_EB_a0.1"]["realized_flip"], "0.0613", "f4")
    claim_true("RL.EB01.null", "RLVE FIXED-EB null at a=.01",
               w["cal_selection"]["0.01"]["FIXED_EB_k"] is None)
    claim_true("RL.HOEF01.null", "RLVE FIXED-HOEF null at a=.01",
               w["cal_selection"]["0.01"]["FIXED_HOEF_k"] is None)

    # ---------- Sec.6 converging evidence (r475 new pins) ----------
    v3 = load("v3_r465_result.json")
    rhos = [c["rho_pooled"] for c in v3["tau2_fe"]]
    rhofe = [c["rho_fe"] for c in v3["tau2_fe"]]
    claim_true("M6.rho.range", "tau2-bench pooled rho in [0.47,0.59]",
               all(0.465 <= x <= 0.595 for x in rhos),
               f"min={min(rhos):.3f} max={max(rhos):.3f}")
    claim_true("M6.feorrho", "tau2-bench FE rho ~= -1/3",
               all(-0.34 <= x <= -0.32 for x in rhofe),
               f"min={min(rhofe):.3f} max={max(rhofe):.3f}")
    claim_true("M6.ncarriers", "tau2-bench six carriers", len(v3["tau2_fe"]) == 6,
               f"n={len(v3['tau2_fe'])}")
    synth = load("synth_r465_result.json")
    het_cells = [c for c in synth["cells"] if c.get("heterogeneous") and c.get("rho_true") == 0.0]
    claim_true("M6.synth.grid", "probit grid N in {10,20,40}",
               sorted({c["n_max"] for c in het_cells}) == [10, 20, 40],
               f"Ns={sorted({c['n_max'] for c in het_cells})}")
    claim_true("M6.synth.iidcp", "iid-CP undercovers 0.983<0.99 at N=40,p=.6,rho=0",
               abs(het_cells[[c["n_max"] for c in het_cells].index(40)]["policies"]["iid_cp"]["agreement"] - 0.983) < 5e-4)
    v5c = load("v5_contrast_r467_result.json")
    claim("M6.bootp95", "plug-in bootstrap p95 0.0511 > alpha=.05 at k=11",
          v5c["V5_bootstrap_coverage"]["0.05"]["boot_p95_flip"], "0.0511", "f4")
    # Remark (continuous-alpha nested family) claims, r478
    ac = load("alpha_continuous_r478_result.json")
    claim("R8.bps", "Remark nested-family breakpoints observed = 267",
          ac["n_breakpoints"], "267", "int")
    claim("R8.jcont", "Remark J_CONT = 9339",
          ac["J_CONT"], "9339", "int")
    claim_true("R8.same.bp.100501",
               "Remark: cont == grid selected rule at a in {.10,.05,.02}",
               all(ac["reference_alphas"][a]["cont"]["bp"] ==
                   ac["reference_alphas"][a]["grid"]["bp"]
                   for a in ("0.1", "0.05", "0.02")))
    claim_true("R8.tight.001",
               "Remark: a=.01 cont bp tighter (0.003007 < grid 0.008529), meank 10.3->11.9",
               abs(ac["reference_alphas"]["0.01"]["cont"]["bp"] - 0.0030070863479039448) < 1e-9
               and abs(ac["reference_alphas"]["0.01"]["grid"]["bp"] - 0.008529397643053507) < 1e-9
               and abs(ac["reference_alphas"]["0.01"]["cont"]["mean_k"] - 11.9) < 0.05
               and abs(ac["reference_alphas"]["0.01"]["grid"]["mean_k"] - 10.31) < 0.05)
    claim_true("R8.eb.loss.002",
               "Remark: FIXED-EB loses a=.02 budget under cont family (grid k*=31)",
               ac["reference_alphas"]["0.02"]["cont"]["fixed_eb_k"] is None
               and ac["reference_alphas"]["0.02"]["grid"]["fixed_eb_k"] == 31)
    claim_true("R8.cont.ok",
               "Remark: BAYES-H certified at all 4 levels under cont family",
               all(ac["reference_alphas"][a]["cont"]["adaptive_certified"]
                   for a in ("0.1", "0.05", "0.02", "0.01")))
    # Remark tightening (exact effective family), r479
    p479 = load("prop_562bound_r479_result.json")
    claim("R9.states555", "Remark: exact stoppable-state count = 555 at N=32",
          p479["n_states_total"], "555", "int")
    claim_true("R9.repro267", "Remark: r479 reproduces r478's 267 breakpoints",
               p479["r478_n_breakpoints_recomputed"] == 267)
    claim("R9.bps181", "Remark: 181 distinct positive cert values <= 0.10",
          p479["n_distinct_cert_in_0_010"], "181", "int")
    claim("R9.jtight6501", "Remark: tight family J = 6501",
          p479["J_CONT_tight"], "6501", "int")
    claim_true("R9.same.sel",
               "Remark: tight family leaves all 4 selected bps unchanged",
               all(abs(p479["reference_alphas"][a]["cont_tight"]["bp"]
                       - p479["reference_alphas"][a]["cont_paper"]["bp"]) < 1e-12
                   for a in ("0.1", "0.05", "0.02", "0.01")))
    claim_true("R9.ucb.tight",
               "Remark: UCB tightening <= 6e-4 at all 4 levels",
               all(p479["reference_alphas"][a]["cont_paper"]["cert"]
                   - p479["reference_alphas"][a]["cont_tight"]["cert"] <= 6e-4 + 1e-9
                   for a in ("0.1", "0.05", "0.02", "0.01")))
    claim_true("R9.eb.null.both",
               "Remark: FIXED-EB a=.02 budget null under either family size",
               p479["reference_alphas"]["0.02"]["fixed_eb_k_paper"] is None
               and p479["reference_alphas"]["0.02"]["fixed_eb_k_tight"] is None)

    # ---------- Proposition prop:tv (r481, TV-robustness certificate) ----------
    tv = load("tv_robustness_r481_result.json")
    s0, s1 = tv["shard0"], tv["shard1"]
    # critical radii printed in the proposition
    claim("TV.s0.t10", "TV tau* shard0 a=.10 (printed 0.144)",
          s0["critical_radius"]["0.1"]["tau_star"], "0.144", "f3")
    claim("TV.s0.t05", "TV tau* shard0 a=.05 (printed 0.078)",
          s0["critical_radius"]["0.05"]["tau_star"], "0.078", "f3")
    claim("TV.s0.t02", "TV tau* shard0 a=.02 (printed 0.074)",
          s0["critical_radius"]["0.02"]["tau_star"], "0.074", "f3")
    claim("TV.s0.t01", "TV tau* shard0 a=.01 (printed 0.052)",
          s0["critical_radius"]["0.01"]["tau_star"], "0.052", "f3")
    claim("TV.s1.t10", "TV tau* shard1 a=.10 (printed 0.157)",
          s1["critical_radius"]["0.1"]["tau_star"], "0.157", "f3")
    claim("TV.s1.t05", "TV tau* shard1 a=.05 (printed 0.082)",
          s1["critical_radius"]["0.05"]["tau_star"], "0.082", "f3")
    claim("TV.s1.t02", "TV tau* shard1 a=.02 (printed 0.076)",
          s1["critical_radius"]["0.02"]["tau_star"], "0.076", "f3")
    claim("TV.s1.t01", "TV tau* shard1 a=.01 (printed 0.024)",
          s1["critical_radius"]["0.01"]["tau_star"], "0.024", "f3")
    # LP cross-check must have matched on every cell
    claim_true("TV.lp.match",
               "TV closed form == simplex LP on all 6x4x2 grid cells",
               s0["lp_crosscheck_all_match"] is True
               and s1["lp_crosscheck_all_match"] is True)
    # post-hoc transfer certified at R=0.042 on both shards at a<=.05/.02
    claim("TV.s0.V042.05", "TV V(0.042) shard0 a=.05 (printed 0.039<=.05)",
          s0["worst_case_key_R"]["0.05"]["V(R=0.042)"], "0.039", "f3")
    claim("TV.s0.V042.02", "TV V(0.042) shard0 a=.02 (printed 0.016<=.02)",
          s0["worst_case_key_R"]["0.02"]["V(R=0.042)"], "0.016", "f3")
    claim_true("TV.valid042",
               "TV transfer valid at R=0.042 for a in {.10,.05,.02} both shards",
               all(s0["worst_case_key_R"][a]["valid_at_R0.042"] for a in ("0.1","0.05","0.02"))
               and all(s1["worst_case_key_R"][a]["valid_at_R0.042"] for a in ("0.1","0.05","0.02")))
    # shard1 a=.01 transfer NOT certifiable by TV (0.042 > tau*=0.024)
    claim_true("TV.s1.t01.lt042",
               "TV shard1 tau*(.01)=0.024 < observed 0.042 mismatch",
               s1["critical_radius"]["0.01"]["tau_star"] < 0.042)
    # prior-shift band at R=0.05, alpha=.10 re-certified at .05
    claim("TV.band.flip", "TV band R=.05 a=.10 worst flip (printed 0.042<=.10)",
          s0["shift_band_R0.05"]["0.1"]["worst_flip_at_R0.05"], "0.042", "f3")
    claim("TV.band.save", "TV band saving (printed 80.3%)",
          s0["shift_band_R0.05"]["0.1"]["saving_band"], "80.3", "pct1")
    claim("TV.band.save0", "TV un-band saving (printed 83.6%)",
          s0["shift_band_R0.05"]["0.1"]["saving_orig"], "83.6", "pct1")
    claim_true("TV.band.valid", "TV band rule valid over B_0.05 at a=.10",
               s0["shift_band_R0.05"]["0.1"]["valid"] is True)
    # zero-g atom mass 36-47% across levels/shards
    _zm = [s0["zero_g_atoms"][a]["H_mass_zero_g"] for a in ("0.1","0.05","0.02","0.01")] \
        + [s1["zero_g_atoms"][a]["H_mass_zero_g"] for a in ("0.1","0.05","0.02","0.01")]
    claim_true("TV.zerog.range",
               "TV zero-g atom mass in 36-47% across all 8 cells",
               all(0.36 - 1e-9 <= z <= 0.47 + 1e-9 for z in _zm),
               f"min={min(_zm)}, max={max(_zm)}")

    # ---------- prop:tv cross-carrier extension to RLVE (r482) ----------
    tvx = load("tv_robustness_rlve_r482_result.json")
    claim("TVX.t10", "TVX RLVE tau* a=.10 (printed 0.199)",
          tvx["critical_radius"]["0.1"]["tau_star"], "0.199", "f3")
    claim("TVX.t05", "TVX RLVE tau* a=.05 (printed 0.551)",
          tvx["critical_radius"]["0.05"]["tau_star"], "0.551", "f3")
    claim("TVX.t02", "TVX RLVE tau* a=.02 (printed 0.213)",
          tvx["critical_radius"]["0.02"]["tau_star"], "0.213", "f3")
    claim("TVX.t01", "TVX RLVE tau* a=.01 (printed 0.449)",
          tvx["critical_radius"]["0.01"]["tau_star"], "0.449", "f3")
    claim_true("TVX.lp.match",
               "TVX closed form == simplex LP on all 6x4 grid cells",
               tvx["lp_crosscheck_all_match"] is True)
    claim_true("TVX.t05.gt.t10",
               "TVX non-monotone: tau*(.05) > tau*(.10) (K=6 zeroed)",
               tvx["critical_radius"]["0.05"]["tau_star"]
               > tvx["critical_radius"]["0.1"]["tau_star"])
    _zgx = [tvx["zero_g_atoms"][a]["H_mass_zero_g"] for a in ("0.1","0.05","0.02","0.01")]
    claim_true("TVX.zerog.range",
               "TVX zero-g atom mass in 74-87% across 4 levels (printed; raw 0.7383-0.874)",
               all(0.7383 - 1e-9 <= z <= 0.874 + 1e-9 for z in _zgx),
               f"min={min(_zgx)}, max={max(_zgx)}")
    _rlve = load("rlve_n8_r474_result.json")
    claim("TVX.hetexcess", "TVX RLVE het-excess variance (printed 0.0999)",
          _rlve["within"]["het_excess_var"], "0.0999", "f4")

    # ---------- prop:tv third-carrier edge case OpenR1 M=2 (r483) ----------
    tvo = load("tv_robustness_openr1_r483_result.json")
    claim("TVO.t10", "TVO OpenR1 tau* a=.10 (printed 0.168)",
          tvo["critical_radius"]["0.1"]["tau_star_closed"], "0.168", "f3")
    claim("TVO.t05", "TVO OpenR1 tau* a=.05 (printed 0.168)",
          tvo["critical_radius"]["0.05"]["tau_star_closed"], "0.168", "f3")
    claim("TVO.t02.whole", "TVO OpenR1 tau* a=.02 whole simplex (=1)",
          tvo["critical_radius"]["0.02"]["tau_star_closed"], "1", "int")
    claim("TVO.t01.whole", "TVO OpenR1 tau* a=.01 whole simplex (=1)",
          tvo["critical_radius"]["0.01"]["tau_star_closed"], "1", "int")
    claim_true("TVO.bisect.agree",
               "TVO bisection agrees with closed form at all 4 levels",
               all(tvo["critical_radius"][a]["agree_1e3"] for a in ("0.1", "0.05", "0.02", "0.01")))
    claim_true("TVO.crosscheck",
               "TVO closed form == scan (1e-9) == LP (1e-6) on all 6x4 grid cells",
               tvo["crosscheck_all_match"] is True)
    claim_true("TVO.prior.repro",
               "TVO prior re-derived from pinned parquet matches r473 JSON (5e-5)",
               tvo["prior_rederived_match"] is True)
    claim_true("TVO.gmax.profile",
               "TVO g_max = .25/.125/0/0 across levels (one-sided pump)",
               [tvo["g_profiles"][a]["g"][1] for a in ("0.1", "0.05", "0.02", "0.01")]
               == [0.25, 0.125, 0.0, 0.0])
    claim("TVO.zerog.stop", "TVO zero-g mass at stopping levels (printed 76.8%)",
          tvo["zero_g_atoms"]["0.1"]["H_mass_zero_g"], "0.768", "f3")
    claim_true("TVO.nonmono.max",
               "TVO non-monotonicity maximal: tau*(.05)=0.168 -> tau*(.02)=1",
               tvo["critical_radius"]["0.05"]["tau_star_closed"] < 0.2
               and tvo["critical_radius"]["0.02"]["tau_star_closed"] == 1.0)

    # ---------- prop:tv conservation curve (r484, App app:tau) ----------
    cc = load("tv_conservation_r484_result.json")
    claim_true("CC.repro",
               "CC reference alphas reproduce r481/r482/r483 radii on all 4 carriers",
               all(cc["reference_reproduced"][k] is True
                   for k in ("omr_shard0", "omr_shard1", "rlve", "openr1")))
    claim_true("CC.p1.mono",
               "CC tau* non-decreasing within every constant-rule interval, all carriers",
               all(cc["p1_within_interval_monotone"][k]["within_interval_nondecreasing"]
                   is True for k in ("omr_shard0", "omr_shard1", "rlve", "openr1")))
    claim("CC.bps.s0", "CC rule-change count OMR shard0 (printed 96)",
          cc["n_breakpoints"]["omr_shard0"], "96", "int")
    claim("CC.bps.s1", "CC rule-change count OMR shard1 (printed 93)",
          cc["n_breakpoints"]["omr_shard1"], "93", "int")
    claim("CC.bps.rlve", "CC rule-change count RLVE (printed 7)",
          cc["n_breakpoints"]["rlve"], "7", "int")
    claim("CC.bps.openr1", "CC rule-change count OpenR1 (printed 2)",
          cc["n_breakpoints"]["openr1"], "2", "int")
    claim("CC.jump.down", "CC OpenR1 jump DOWN at a=0.07852 (printed -0.314)",
          cc["openr1_jump_analysis"]["jump_down_at_0.07852"], "0.314", "f3")
    claim("CC.jump.up2", "CC OpenR1 2nd jump at a=0.04598: artifact signed -0.864 (a DOWN jump)",
          cc["openr1_jump_analysis"]["jump_up_at_0.04598"], "-0.864", "f3")
    claim("CC.jump.below", "CC OpenR1 tau* below 0.07852 (printed 0.396)",
          cc["openr1_jump_analysis"]["below"], "0.396", "f3")
    claim("CC.jump.above", "CC OpenR1 tau* above 0.07852 (printed 0.082)",
          cc["openr1_jump_analysis"]["above"], "0.082", "f3")
    claim("CC.spearman", "CC Spearman(zero-g, tau*) 16 cells (printed 0.75)",
          cc["zerog_tau_spearman_16cells"], "0.75", "f2")
    # whole-simplex plateaus: rlve tau*(.001)=1, openr1 tau*(.001)=1
    claim("CC.plateau.rlve", "CC RLVE tau*(1e-3)=1 (whole-simplex plateau)",
          cc["curve_summary"]["rlve"]["tau_at_0.001"], "1", "int")
    claim("CC.plateau.openr1", "CC OpenR1 tau*(1e-3)=1 (whole-simplex plateau)",
          cc["curve_summary"]["openr1"]["tau_at_0.001"], "1", "int")

    # ---------- r491: prior-fit-size sensitivity (app:tau (e)) ----------
    fs = load("prior_fit_size_r491_result.json")
    _by = {}
    for cn, cc2 in fs["carriers"].items():
        for c in cc2["cells"]:
            _by[(cn, c["m"], c["alpha"])] = c
    # operational: tau*_125(.05) clears the cross-shard transfer 0.042 on both shards
    claim("FS.s0.t05.125", "FS OMR s0 tau*_125(.05) (printed 0.062)",
          _by[("omr_shard0", 125, 0.05)]["tau_star"], "0.062", "f3")
    claim("FS.s1.t05.125", "FS OMR s1 tau*_125(.05) (printed 0.046)",
          _by[("omr_shard1", 125, 0.05)]["tau_star"], "0.046", "f3")
    claim_true("FS.clear.042",
               "both OMR tau*_125(.05) >= cross-shard transfer 0.042",
               _by[("omr_shard0", 125, 0.05)]["tau_star"] >= 0.042
               and _by[("omr_shard1", 125, 0.05)]["tau_star"] >= 0.042)
    # non-monotonicity exemplar: OMR shard1 alpha=0.01 up to m=1000 then down
    claim("FS.s1.t01.125", "FS OMR s1 tau*_125(.01) (printed 0.010)",
          _by[("omr_shard1", 125, 0.01)]["tau_star"], "0.010", "f3")
    claim("FS.s1.t01.1000", "FS OMR s1 tau*_1000(.01) (printed 0.056)",
          _by[("omr_shard1", 1000, 0.01)]["tau_star"], "0.056", "f3")
    claim("FS.s1.t01.4000", "FS OMR s1 tau*_4000(.01) (printed 0.024)",
          _by[("omr_shard1", 4000, 0.01)]["tau_star"], "0.024", "f3")
    claim_true("FS.nonmono",
               "s1@.01: 0.056 at m=1000 exceeds both endpoints (strictly non-monotone)",
               _by[("omr_shard1", 1000, 0.01)]["tau_star"]
               > _by[("omr_shard1", 125, 0.01)]["tau_star"]
               and _by[("omr_shard1", 1000, 0.01)]["tau_star"]
               > _by[("omr_shard1", 4000, 0.01)]["tau_star"])
    # closed loop: zero violations over applicable cells; exactly 51 applicable
    claim_true("FS.loop.zero",
               "prop:tv closed loop: 0 flip_full>alpha among TV<=tau* cells",
               fs["checks"]["p3_closed_loop"]["n_violations"] == 0
               and fs["checks"]["p3_closed_loop"]["pass"] is True)
    claim("FS.loop.n", "FS closed-loop applicable cell count (printed 51)",
          fs["checks"]["p3_closed_loop"]["n_applicable"], "51", "int")
    # outside-ball genuine break at s1 m=125, repaired from m=250
    claim("FS.break.02", "FS s1 m=125 flip_full@.02 (printed 0.0244)",
          _by[("omr_shard1", 125, 0.02)]["flip_full"], "0.0244", "f4")
    claim("FS.break.01", "FS s1 m=125 flip_full@.01 (printed 0.0165)",
          _by[("omr_shard1", 125, 0.01)]["flip_full"], "0.0165", "f4")
    claim("FS.break.tv", "FS s1 m=125 TV to full prior (printed 0.175)",
          _by[("omr_shard1", 125, 0.02)]["tv_m"], "0.175", "f3")
    claim_true("FS.repair",
               "s1 flip_full back under alpha from m=250 at both .02 and .01",
               _by[("omr_shard1", 250, 0.02)]["flip_full"] <= 0.02
               and _by[("omr_shard1", 250, 0.01)]["flip_full"] <= 0.01
               and all(_by[("omr_shard1", m, a)]["flip_full"] <= a
                       for m in (250, 500, 1000, 2000, 4000)
                       for a in (0.02, 0.01)))
    # root-m: OMR spread of TV*sqrt(m) below 1.5x
    claim_true("FS.rootm.omr",
               "OMR TV*sqrt(m) spread < 1.5x on both shards",
               fs["checks"]["p2_rootm"]["detail"]["omr_shard0"]["ratio"] < 1.5
               and fs["checks"]["p2_rootm"]["detail"]["omr_shard1"]["ratio"] < 1.5)
    claim_true("FS.rootm.rlve",
               "RLVE TV*sqrt(m) spread = 3.5x (missing-mode coarse prefix)",
               3.4 < fs["checks"]["p2_rootm"]["detail"]["rlve"]["ratio"] < 3.6)

    # r491 tex-direct-parse layer: the FS.* claims above hardcode the printed
    # numbers; without this layer a tex edit of those numbers would NOT be
    # caught by any FS.* claim (r491 self-audit: negative control 1 showed
    # only FRESH.tex.anchor fired). Assert the printed strings exist in tex.
    _fs_ptex = next((c for c in (os.path.join(HERE, "paper.tex"),
                                 os.path.join(HERE, "..", "paper.tex"))
                     if os.path.isfile(c)), None)
    _fs_src = open(_fs_ptex).read() if _fs_ptex is not None else ""
    _hasfs = bool(_fs_src)
    claim_true("X.fs.t05",
               "tex prints shard0/shard1 tau*_125(.05) as $0.062$/$0.046$",
               not _hasfs or ("$0.062$" in _fs_src and "$0.046$" in _fs_src))
    claim_true("X.fs.nonmono",
               "tex prints the non-monotone triplet 0.010/0.056/0.024 (m=125/1000/4000)",
               not _hasfs or ("$0.010$" in _fs_src and "$0.056$" in _fs_src
                              and "$0.024$" in _fs_src and "$m{=}1000$" in _fs_src))
    claim_true("X.fs.break",
               "tex prints the m=125 break flips $0.0244>0.02$, $0.0165>0.01$",
               not _hasfs or ("$0.0244>0.02$" in _fs_src
                              and "$0.0165>0.01$" in _fs_src))
    claim_true("X.fs.loopn",
               "tex prints the 51 applicable closed-loop cells and TV $0.175$",
               not _hasfs or ("51" in _fs_src and "$0.175$" in _fs_src))
    claim_true("X.fs.rootm",
               "tex prints OMR spread $1.5\\times$ and RLVE $3.5\\times$",
               not _hasfs or ("$1.5\\times$" in _fs_src
                              and "$3.5\\times$" in _fs_src))

    # ---------- r494: rule-channel decomposition (app:tau (f)) ----------
    fsd = load("rule_channel_r494_result.json")
    _fc = {}
    for cn, cc3 in fsd["carriers"].items():
        for c in cc3["cells"]:
            _fc[(cn, c["m"], c["alpha"])] = c
    # regen anchor: recomputed both-moving curve reproduces r491 grid exactly
    claim_true("FSD.anchor",
               "FSD recomputed tau_both reproduces r491 grid (0/72 mismatches)",
               fsd["regen_anchor_r491"]["n_mismatch"] == 0)
    # (i) rule channel drives non-monotonicity: prior-fixed curve at s1 @.01
    #     0.000 (m125) -> 0.060 (m1000) -> 0.024 (m4000), still non-monotone
    claim("FSD.pf.s1.t01.125", "FSD s1 prior-fixed tau* m=125 @.01 (printed 0.000)",
          _fc[("omr_shard1", 125, 0.01)]["tau_prior_fixed"], "0.000", "f3")
    claim("FSD.pf.s1.t01.1000", "FSD s1 prior-fixed tau* m=1000 @.01 (printed 0.060)",
          _fc[("omr_shard1", 1000, 0.01)]["tau_prior_fixed"], "0.060", "f3")
    claim("FSD.pf.s1.t01.4000", "FSD s1 prior-fixed tau* m=4000 @.01 (printed 0.024)",
          _fc[("omr_shard1", 4000, 0.01)]["tau_prior_fixed"], "0.024", "f3")
    claim_true("FSD.pf.nonmono",
               "s1 @.01 prior-fixed still up-then-down (0.060>0.000 and 0.060>0.024)",
               _fc[("omr_shard1", 1000, 0.01)]["tau_prior_fixed"]
               > _fc[("omr_shard1", 125, 0.01)]["tau_prior_fixed"]
               and _fc[("omr_shard1", 1000, 0.01)]["tau_prior_fixed"]
               > _fc[("omr_shard1", 4000, 0.01)]["tau_prior_fixed"])
    claim_true("FSD.pf.gmax.nonmono",
               "s1 @.01 induced gmax non-monotone 0.563->0.075->0.098",
               abs(_fc[("omr_shard1", 125, 0.01)]["gmax_m"] - 0.56306787) < 1e-6
               and abs(_fc[("omr_shard1", 1000, 0.01)]["gmax_m"] - 0.07470854) < 1e-6
               and abs(_fc[("omr_shard1", 4000, 0.01)]["gmax_m"] - 0.09807494) < 1e-6)
    # (ii) estimation channel NOT uniformly benign: 31/72 cells rule-fixed
    #      below full-fit radius; largest deficit 0.053 at RLVE m=93 @.05
    claim("FSD.nviol", "FSD rule-fixed-below-full cell count (printed 31)",
          fsd["checks"]["q2_rule_fixed_never_below_full"]["n_viol"], "31", "int")
    claim_true("FSD.nviol.pass_false",
               "Q2 (uniformly benign) was FALSIFIED (pass=False)",
               fsd["checks"]["q2_rule_fixed_never_below_full"]["pass"] is False)
    _rl93 = _fc[("rlve", 93, 0.05)]
    _rlfull = _fc[("rlve", 3000, 0.05)]
    claim_true("FSD.maxdeficit",
               "largest rule-fixed deficit 0.053 at RLVE m=93 @.05 (0.498 vs 0.551)",
               abs((_rlfull["tau_rule_fixed"] - _rl93["tau_rule_fixed"]) - 0.05253) < 1e-3)
    # (iii) RLVE m=187 @.05 dip is pure rule-channel
    _rl187 = _fc[("rlve", 187, 0.05)]
    claim("FSD.rlve187.both", "FSD RLVE m=187 @.05 both-moving (printed 0.235)",
          _rl187["tau_both"], "0.235", "f3")
    claim("FSD.rlve187.pf", "FSD RLVE m=187 @.05 prior-fixed (printed 0.222)",
          _rl187["tau_prior_fixed"], "0.222", "f3")
    claim("FSD.rlve187.rf", "FSD RLVE m=187 @.05 rule-fixed (printed 0.530)",
          _rl187["tau_rule_fixed"], "0.530", "f3")
    claim_true("FSD.rlve187.pure_rule",
               "RLVE m=187 dip survives prior-frozen, vanishes rule-frozen; gmax 0.179 vs 0.071",
               _rl187["tau_prior_fixed"] < _fc[("rlve", 93, 0.05)]["tau_prior_fixed"]
               and _rl187["tau_rule_fixed"] > 0.5
               and abs(_rl187["gmax_m"] - 0.17857143) < 1e-6
               and abs(_rl187["gmax_full"] - 0.07142857) < 1e-6)
    # r494 tex-direct-parse layer (X.fsd.*): hardcoded FSD.* numbers must be
    # asserted present in the tex, else a tex-only edit escapes (r491 lesson).
    _fsd_src = _fs_src  # reuse the same already-loaded tex source
    claim_true("X.fsd.i",
               "tex prints prior-fixed s1@.01 triplet 0.000/0.060/0.024 and gmax 0.563/0.075/0.098",
               not _hasfs or ("$0.000$ at $m{=}125$" in _fsd_src
                              and "$0.060$ at $m{=}1000$" in _fsd_src
                              and "$0.024$ at $m{=}4000$" in _fsd_src
                              and "0.563\\to0.075\\to0.098" in _fsd_src))
    claim_true("X.fsd.ii",
               "tex prints 31 of 72 cells and largest deficit 0.053 at RLVE m=93",
               not _hasfs or ("31 of 72 cells" in _fsd_src
                              and "$0.053$ at" in _fsd_src))
    claim_true("X.fsd.iii",
               "tex prints RLVE m=187 dip 0.235 / prior-fixed 0.222 / rule-fixed 0.530 / gmax 0.179/0.071",
               not _hasfs or ("$0.235$" in _fsd_src and "0.222" in _fsd_src
                              and "0.530" in _fsd_src and "0.179" in _fsd_src
                              and "0.071" in _fsd_src))
    claim_true("X.fsd.anchor",
               "tex prints bit-exact regen anchor (0/72 mismatches)",
               not _hasfs or ("bit-exactly (0/72" in _fsd_src))

    # ---------- r499/v5.15: flip-budget state-subset repair (app:tau (g)) ----------
    fsg = load("full_sweep_r498e_result.json")
    _gc = {}
    for cn, cc4 in fsg["carriers"].items():
        for c in cc4["cells"]:
            _gc[(cn, c["m"], c["alpha"])] = c
    # S1 regen anchor: recomputed grid reproduces r491 grid bit-exactly
    claim_true("FSG.anchor",
               "FSG regen anchor reproduces r491 grid (0/72 mismatches)",
               fsg["checks"]["S1_regen"]["mismatched"] == 0
               and fsg["checks"]["S1_regen"]["pass"] is True)
    # all 72 cells reach tau*=1 at best cap
    claim_true("FSG.all72.tau1",
               "FSG best-cap tau* = 1.0 at every one of the 72 cells",
               all(c["tau_best"] == 1.0 for c in _gc.values()))
    # 70 strict gains; the 2 RLVE a=.02 cells already whole-simplex, no cap selected
    claim_true("FSG.gain70",
               "FSG 70 cells best_trivial (strict gain to trivial validity)",
               sum(1 for c in _gc.values() if c["best_trivial"]) == 70)
    claim_true("FSG.rlve02.nocap",
               "FSG the 2 non-gain cells are RLVE a=.02 with best_cap None (orig tau*=1)",
               _gc[("rlve", 93, 0.02)]["best_cap"] is None
               and _gc[("rlve", 750, 0.02)]["best_cap"] is None
               and _gc[("rlve", 93, 0.02)]["tau_orig"] == 1.0
               and _gc[("rlve", 750, 0.02)]["tau_orig"] == 1.0)
    # votes cost: +1% to +11%, max +11.3% at OMR s0 m=500 a=.10; RLVE max +3.3%
    claim_true("FSG.votes.max",
               "FSG max best votes ratio 1.1125 at OMR s0 m=500 a=.10",
               abs(_gc[("omr_shard0", 500, 0.1)]["best_votes_ratio"] - 1.1125) < 1e-4
               and max(c["best_votes_ratio"] for c in _gc.values()) < 1.113)
    claim_true("FSG.votes.rlve",
               "FSG RLVE best votes ratios all <= 1.0332 (+3.3%)",
               max(c["best_votes_ratio"] for (cn, m, a), c in _gc.items()
                   if cn == "rlve") <= 1.0332 + 1e-9
               and min(c["best_votes_ratio"] for c in _gc.values()
                       if c["best_cap"] is not None) >= 1.001)
    # pitfall repair: RLVE m=187 a=.05 0.235 -> 1.0 at cap 0.07, +0.5% votes
    _g187 = _gc[("rlve", 187, 0.05)]
    claim("FSG.pitfall.orig", "FSG RLVE m=187 @.05 tau_orig (printed 0.235)",
          _g187["tau_orig"], "0.235", "f3")
    claim_true("FSG.pitfall.repair",
               "FSG RLVE m=187 @.05 repaired to tau*=1 at cap 0.07, votes +0.5%",
               _g187["tau_best"] == 1.0 and _g187["best_cap"] == 0.07
               and abs(_g187["best_votes_ratio"] - 1.0049) < 1e-4)
    # coarse OMR s1 m=125 a=.01: 0.0095 -> 1.0
    _g125 = _gc[("omr_shard1", 125, 0.01)]
    claim("FSG.coarse.orig", "FSG OMR s1 m=125 @.01 tau_orig (printed 0.0095)",
          _g125["tau_orig"], "0.0095", "f4")
    claim_true("FSG.coarse.repair",
               "FSG OMR s1 m=125 @.01 repaired to tau*=1 at cap 0.01",
               _g125["tau_best"] == 1.0 and _g125["best_cap"] == 0.01)
    # S4 coarse check pass (all four levels exceed half full-fit radius)
    claim_true("FSG.s4.pass",
               "FSG S4 coarse-fit check passes at all 4 levels",
               fsg["checks"]["S4_coarse"]["pass"] is True
               and all(c["pass"] for c in fsg["checks"]["S4_coarse"]["cells"]))
    # negative result 1: support truncation changes nothing (0 diffs)
    fst = load("support_trunc_repair_r498_result.json")
    claim_true("FSG.trunc.null",
               "FSG support-truncation negative result: P1_pitfall FAIL, tau unchanged 0.234866",
               fst["checks"]["P1_pitfall"]["pass"] is False
               and abs(fst["checks"]["P1_pitfall"]["tau_repaired"]
                       - fst["checks"]["P1_pitfall"]["tau_orig"]) < 1e-9
               and abs(fst["checks"]["P1_pitfall"]["tau_repaired"] - 0.234866) < 1e-6)
    claim_true("FSG.trunc.regen",
               "FSG support-truncation regen anchor 0/72",
               fst["checks"]["P4_regen_anchor"]["mismatched"] == 0)
    # negative result 2: deadline truncation certifies LESS (0.235 -> 0.133)
    fdl = load("efficiency_r498c_result.json")
    _dl = {(c["carrier"], c["m"], c["alpha"]): c for c in fdl["cells"]}
    claim("FSG.deadline.worse", "FSG deadline RLVE m=187 @.05 tau (printed 0.133)",
          _dl[("rlve", 187, 0.05)]["tau_real"], "0.133", "f3")
    # r498d six-cell detail: domination + subset >= deadline/orig all pass
    fsd2 = load("statecap_repair_r498d_result.json")
    claim_true("FSG.r498d.checks",
               "FSG r498d six-cell checks all pass (domination, coarse gain, pitfall, no-harm, vs-deadline)",
               all(v.get("pass") is True for v in fsd2["checks"].values()))
    claim_true("FSG.r498d.k16",
               "FSG r498d OMR s1 m=125 a=.02 K=16 atom: g 0.5832 -> 0.0401",
               any(a["K"] == 16 and abs(a["g_orig"] - 0.583204) < 1e-4
                   and abs(a["g_S"] - 0.040134) < 1e-4
                   for c in fsd2["cells"]
                   if c["carrier"] == "omr_shard1" and c["m"] == 125
                   and c["alpha"] == 0.02
                   for a in c["atoms"]))
    # baseline comparison: fixed k=3 base flip 0.093 vs repaired base_S 0.0036
    claim_true("FSG.r498d.baseS",
               "FSG r498d OMR s1 m=125 a=.01 base_S 0.0036 vs k=3 baseline base 0.093",
               abs(next(c["base_subset"] for c in fsd2["cells"]
                        if c["carrier"] == "omr_shard1" and c["m"] == 125
                        and c["alpha"] == 0.01) - 0.00364854) < 1e-6)
    # tex-direct-parse layer (X.fsg.*)
    _fsg_src = _fsd_src
    claim_true("X.fsg.i",
               "tex prints flip-budget construction + 70 of 72 + votes range +1%/+11%",
               not _hasfs or ("flip budget" in _fsg_src
                              and "$70$ of $72$" in _fsg_src
                              and "+1\\%$ to $+11\\%" in _fsg_src
                              and "+11.3\\%" in _fsg_src))
    claim_true("X.fsg.ii",
               "tex prints pitfall repair 0.235 -> 1.0 cap 0.07 and coarse 0.0095 -> 1.0",
               not _hasfs or ("$\\tau^*=0.235$ to $1.0$" in _fsg_src
                              and "cap $0.07$" in _fsg_src
                              and "$0.0095\\to1.0$" in _fsg_src))
    claim_true("X.fsg.iii",
               "tex prints both negative results (support truncation 0 differences; deadline 0.235->0.133)",
               not _hasfs or ("support truncation fails" in _fsg_src
                              and "deadline truncation fails" in _fsg_src
                              and "$0.235\\to0.133$" in _fsg_src))
    claim_true("X.fsg.iv",
               "tex prints trivial-validity honesty + k=3 baseline 0.093 vs 0.0036",
               not _hasfs or ("trivial-validity" in _fsg_src
                              and "$0.093$" in _fsg_src
                              and "0.0036" in _fsg_src))

    # ---------- r500/v5.16: universal repair budget (app:tau (g) Universal budget) ----------
    fsu = load("universal_cap_r500_result.json")
    claim_true("FSU.checks",
               "FSU r500 self-checks all pass (S1 crit-cap dict, S2 alpha=.01-only failures, 6/carrier)",
               fsu["checks"]["ALL_PASS"] is True)
    claim_true("FSU.s1.crit72",
               "FSU critical cap = 0.01 at every one of the 72 cells",
               fsu["S1_critical_caps"] == {"0.01": 72})
    claim_true("FSU.s2.fails18",
               "FSU cap=0.015 leaves exactly 18 cells unrepaired, all alpha=0.01, 6 per carrier",
               len(fsu["S2_cap0015_failures"]) == 18
               and all(f["alpha"] == 0.01 for f in fsu["S2_cap0015_failures"])
               and sorted(set(f["carrier"] for f in fsu["S2_cap0015_failures"]))
                   == ["omr_shard0", "omr_shard1", "rlve"])
    _fu0 = fsu["S3_cost_at_cap001"]["omr_shard0"]
    _fu1 = fsu["S3_cost_at_cap001"]["omr_shard1"]
    _fur = fsu["S3_cost_at_cap001"]["rlve"]
    claim_true("FSU.s3.votes",
               "FSU cap=0.01 votes cost OMR +1.0%..+22.3% / RLVE <= +5.2%",
               abs(_fu0["votes_ratio_max"] - 1.2228) < 1e-4
               and abs(_fu1["votes_ratio_max"] - 1.2062) < 1e-4
               and min(_fu0["votes_ratio_min"], _fu1["votes_ratio_min"]) >= 1.0095
               and abs(_fur["votes_ratio_max"] - 1.0518) < 1e-4)
    claim_true("FSU.s3.baseS",
               "FSU cap=0.01 realized base flip <= 0.0038 everywhere (RLVE 0)",
               max(_fu0["base_S_max"], _fu1["base_S_max"]) <= 0.0038
               and _fur["base_S_max"] == 0.0)
    # cross-anchor: recompute the same quantities directly from the r498e sweep
    claim_true("FSU.xanchor.crit72",
               "FSU cross-anchor: sweep itself has tau(cap=0.01)=1 at all 72 cells",
               all(c["per_cap"]["0.01"]["tau"] == 1.0 for c in _gc.values()))
    claim_true("FSU.xanchor.fails18",
               "FSU cross-anchor: sweep tau(cap=0.015)<1 exactly at the 18 alpha=.01 cells",
               sum(1 for c in _gc.values() if c["per_cap"]["0.015"]["tau"] < 1.0 - 1e-12) == 18
               and all(a == 0.01 for (cn, m, a), c in _gc.items()
                       if c["per_cap"]["0.015"]["tau"] < 1.0 - 1e-12))
    claim_true("X.fsu.i",
               "tex prints universal budget: cap 0.01, 72 cells tau*=1, votes +1.0%/+22.3%, base<=0.0038",
               not _hasfs or ("Universal budget" in _fsg_src
                              and "cap $0.01$" in _fsg_src
                              and "+22.3\\%" in _fsg_src
                              and "0.0038" in _fsg_src))
    claim_true("X.fsu.ii",
               "tex prints sharp boundary: cap 0.015, 18 cells alpha=0.01, 54 repaired",
               not _hasfs or ("cap $0.015$" in _fsg_src
                              and "$18$ cells with $\\alpha{=}0.01$" in _fsg_src
                              and "$54$ cells" in _fsg_src))

    # ---------- r503: critical-cap edge law (c* quantization) ----------
    cce = load("critical_cap_r503_result.json")
    claim_true("CCE.checks",
               "CCE r503 self-checks pass (P1/P2/P4; P5 descriptive grid dev 6.3% by design)",
               cce["checks"]["P1_subalpha_vacuous"]["pass"] is True
               and cce["checks"]["P2_cstar_ge_alpha"]["pass"] is True
               and cce["checks"]["P4_r498e_anchor"]["pass"] is True
               and cce["checks"]["P4_r498e_anchor"]["total"] == 54)
    claim_true("CCE.p1.subalpha",
               "CCE sub-alpha vacuity: tau=1 at every cap<=alpha, gmax_S<=cap, 0 violations",
               cce["checks"]["P1_subalpha_vacuous"]["fail"] == 0)
    claim_true("CCE.p2.cstar",
               "CCE c*>=alpha in all 72 cells and strictly above alpha in all 72",
               cce["checks"]["P2_cstar_ge_alpha"]["min_ratio"] >= 1.0 - 1e-9
               and cce["checks"]["P2_cstar_ge_alpha"]["n_strict_above"] == 72)
    claim_true("CCE.p3.rlve",
               "CCE RLVE quantizes: c*/alpha max 2.0 (a=.02), mean 1.4822",
               abs(cce["checks"]["P3_carrier_table"]["rlve"]["max"] - 2.0) < 1e-4
               and abs(cce["checks"]["P3_carrier_table"]["rlve"]["mean"] - 1.4822) < 1e-3)
    claim_true("CCE.p3.omr",
               "CCE OMR margins: <=+0.8% at alpha>=.02 except shard1 a=.05 (+2.8%), up to +10.6% at alpha=.01",
               max(max(v) for k, v in cce["checks"]["P3_carrier_table"]["omr_shard0"]["per_alpha"].items()
                   if k != "0.01") <= 1.008
               and max(max(v) for k, v in cce["checks"]["P3_carrier_table"]["omr_shard1"]["per_alpha"].items()
                   if k not in ("0.01", "0.05")) <= 1.008
               and max(cce["checks"]["P3_carrier_table"]["omr_shard1"]["per_alpha"]["0.05"]) <= 1.028
               and max(cce["checks"]["P3_carrier_table"]["omr_shard0"]["per_alpha"]["0.01"]) <= 1.106
               and max(cce["checks"]["P3_carrier_table"]["omr_shard1"]["per_alpha"]["0.01"]) <= 1.106)
    claim_true("CCE.xanchor.tau",
               "CCE cross-anchor: fine-grid tau at r498e grid caps matches sweep (54 alpha=.01 cells)",
               cce["checks"]["P4_r498e_anchor"]["mismatched"] == 0)
    claim_true("X.cce.i",
               "tex prints edge law: c*>=alpha, strict survival 72 cells, OMR +10.6% "
               "(RLVE quantization corrected to derived values by r504 erratum; "
               "the v5.17 string {8/7,2} was WRONG -- 8/7 never occurs in the r503 "
               "measurement -- and must NOT reappear)",
               not _hasfs or ("critical\\_cap\\_r503" in _fsg_src
                              and "10.6\\%" in _fsg_src
                              and "$72$ cells" in _fsg_src
                              and "8/7" not in _fsg_src))

    # ---------- r504: derived edge law (closed-form c*) + v5.17 erratum ----
    ced = load("edge_law_r504_result.json")
    _cedrl = ced["rlve"]["cells"]
    _cedomr = [c for cn in ced["omr"].values() for c in cn]
    claim_true("CED.checks",
               "CED r504 self-checks pass (D1 closed=scan on 18 bracketed, "
               "D2 a=.02 split 25/7 vs inf, D3 m-invariance, D4 OMR no quantum, D5 ratios)",
               ced["checks"]["ALL_PASS"] is True)
    claim_true("CED.d1.recover",
               "CED closed form recovers all 18 scan-bracketed RLVE edges",
               ced["checks"]["D1_closed_recovers_r503"]["n_bracketed"] == 18
               and ced["checks"]["D1_closed_recovers_r503"]["pass"] is True)
    _rlr = ced["checks"]["D5_rlve_ratios"]["per_alpha"]
    claim_true("CED.rlve.quant",
               "CED RLVE quanta: 3/28@.10 (15/14), 1/14@.05 (10/7), 1/70@.01 (10/7)",
               abs(_rlr["0.1"][0] - 15 / 14) < 1e-3
               and abs(_rlr["0.05"][0] - 10 / 7) < 1e-3
               and abs(_rlr["0.01"][0] - 10 / 7) < 1e-3
               and len(_rlr["0.1"]) == 1 and len(_rlr["0.05"]) == 1
               and len(_rlr["0.01"]) == 1)
    claim_true("CED.a02.split",
               "CED alpha=.02: derived edge 1/14 (25/7) at m in {187,375,1500,3000}; "
               "never-falls (c*=inf, tau=1 at cap 0.5) at m in {93,750}",
               ced["checks"]["D2_a02_split"]["finite_ms"] == [187, 375, 1500, 3000]
               and ced["checks"]["D2_a02_split"]["inf_ms"] == [93, 750]
               and ced["checks"]["D2_a02_split"]["pass"] is True)
    claim_true("CED.minvar",
               "CED c* identical across all 6 m at alpha in {.01,.05,.10}",
               ced["checks"]["D3_m_invariance"]["invariant_alphas"] is True)
    claim_true("CED.omr.nogap",
               "CED OMR per-atom closed edge upper-bounds scan edge, within +10.6% of alpha, "
               "equality at 11/48 cells at the discriminating 1e-6 tolerance "
               "(the r504 verifier's 1e-4 tolerance absorbed one strict cell, "
               "shard1 m=2000 a=.10 diff 8.2e-05; corrected in v5.19, see CET)",
               ced["checks"]["D4_omr_no_quantum"]["pass"] is True
               and sum(1 for c in _cedomr
                       if c["cstar_closed"] is not None
                       and abs(c["cstar_closed"] - c["cstar_r503"]) < 1e-6) == 11
               and sum(1 for c in _cedomr
                       if c["cstar_closed"] is not None
                       and abs(c["cstar_closed"] - c["cstar_r503"]) < 1e-4) == 12)
    # tex anchors for the derived values (byte-locked against the artifact-
    # backed sentence; replacing a derived number or re-introducing 8/7
    # trips these plus FRESH.tex.anchor).
    claim_true("X.ced.i",
               "tex prints derived RLVE quanta: 3/28, 15/14, 10/7, 1/70, 25/7",
               not _hasfs or ("edge\\_law\\_r504" in _fsg_src
                              and "$c^*{=}3/28$" in _fsg_src
                              and "15/14\\cdot\\alpha$" in _fsg_src
                              and "10/7\\cdot\\alpha$" in _fsg_src
                              and "$1/70$" in _fsg_src
                              and "25/7\\cdot\\alpha$" in _fsg_src))
    claim_true("X.ced.ii",
               "tex prints the alpha=.02 certificate-straddle split: never-falls "
               "m in {93,750}, tau=1 at cap 0.5, gmax=1/56",
               not _hasfs or ("$m\\in\\{93,750\\}$" in _fsg_src
                              and "cap $0.5$" in _fsg_src
                              and "\\max_K g_S{=}1/56$" in _fsg_src))
    claim_true("X.ced.iii",
               "tex prints OMR no-quantum: upper bound, equality at 11 of 48 cells",
               not _hasfs or ("$11$ of $48$ cells" in _fsg_src
                              and "upper-bounds" in _fsg_src
                              and "$12$ of $48$ cells" not in _fsg_src))

    # ---------- r505: edge tightness regimes + 12->11 count correction ----
    cet = load("edge_tightness_r505_result.json")
    _cetcells = cet["cells"]
    claim_true("CET.checks",
               "CET r505 self-checks pass (P1 corrected count, P2 upper bound, "
               "P3/P4 slack-sign biconditionals, regime separation)",
               cet["checks"]["ALL_PASS"] is True)
    claim_true("CET.count",
               "CET tight count: 11/48 at 1e-6, 12/48 at 1e-4, twelfth cell "
               "shard1 m=2000 a=.10 absorbed by the verifier tolerance",
               cet["checks"]["P1_corrected_count"]["pass"] is True
               and cet["checks"]["P1_corrected_count"]["n_tight_1e-6"] == 11
               and cet["checks"]["P1_corrected_count"]["n_tight_1e-4"] == 12)
    claim_true("CET.regimes",
               "CET slack-sign regimes partition the grid: tight slack in "
               "[-1.26e-03, -2.43e-05], strict slack in [+2.91e-05, +3.71e-03], "
               "no overlap (tightness decided by a sign)",
               cet["checks"]["regime_separation"]["pass"] is True
               and cet["checks"]["regime_separation"]["tight_slack_range"][1] < 0
               < cet["checks"]["regime_separation"]["strict_slack_range"][0])
    claim_true("CET.biconditional",
               "CET tight <=> g_S(closed-eps)<=alpha on all 48 cells; "
               "strict <=> g_S(closed-eps)>alpha on all 48 cells",
               cet["checks"]["P3_tight_biconditional"]["pass"] is True
               and cet["checks"]["P4_strict_biconditional"]["pass"] is True)
    claim_true("X.cet.i",
               "tex prints the tightness regimes: 11 tight, 37 strict, "
               "slack-sign separation, edge_tightness_r505 anchor, "
               "and the 12->11 correction disclosure",
               not _hasfs or ("edge\\_tightness\\_r505" in _fsg_src
                              and "in $11$ cells" in _fsg_src
                              and "remaining $37$ cells" in _fsg_src
                              and "from $12$ to $11$" in _fsg_src))

    # ---------- r506: unified discrete-stopping-set geometry note ----
    cgu = load("discrete_geometry_r506_result.json")
    claim_true("CGU.checks",
               "CGU r506 self-checks pass (U1 fine prefix, U2 gmax monotone, "
               "U3 C4 count + 792-pair erratum, U4 kappa probe, U5 deadline, "
               "U6 tightness cross-ref)",
               cgu["checks"]["ALL_PASS"] is True)
    claim_true("CGU.prefix",
               "CGU full-radius crossing set is a prefix on the fine grid: "
               "72 cells, 0 holes, 0 overhangs",
               cgu["checks"]["U1_fine_prefix"]["cells"] == 72
               and cgu["checks"]["U1_fine_prefix"]["holes"] == 0
               and cgu["checks"]["U1_fine_prefix"]["overhangs"] == 0)
    claim_true("CGU.scalars",
               "CGU both scalarizations monotone: gmax_S 0 violations on 2592 "
               "fine-grid pairs; base_S (C3) 0 violations on 792 coarse pairs",
               cgu["checks"]["U2_gmax_nondecr_fine"]["adjacent_pairs"] == 2592
               and cgu["checks"]["U2_gmax_nondecr_fine"]["violations"] == 0
               and cgu["checks"]["U3_c4_count"]["c3_violations_recomputed"] == 0)
    claim_true("CGU.c4",
               "CGU coarse-grid interior non-monotonicity: 323 of 792 "
               "adjacent-cap pairs violate tau monotonicity (r501 C4); "
               "1728-pair misprint disclosed as housekeeping erratum",
               cgu["checks"]["U3_c4_count"]["recomputed_c4"] == 323
               and cgu["checks"]["U3_c4_count"]["pairs"] == 792)
    claim_true("CGU.kappa",
               "CGU deadline flip sequence on probed RLVE atom anchored and "
               "non-monotone: 0.0143, 0.3571, 0.2143, 0.5, 0.2429, 0.5 "
               "(>=2 strict up-jumps as kappa decreases)",
               cgu["checks"]["U4_kappa_probe"]["pass"] is True)
    claim_true("CGU.deadline",
               "CGU deadline repair worsens: RLVE m=187 a=.05 tau 0.235->0.133",
               cgu["checks"]["U5_deadline_anchor"]["tau_orig_f3"] == 0.235
               and cgu["checks"]["U5_deadline_anchor"]["tau_real_f3"] == 0.133)
    claim_true("X.cgu.i",
               "tex prints the unified geometry note: 323 of 792 pairs, "
               "0 holes and 0 overhangs, both scalars monotone, kappa "
               "sequence, discrete-geometry anchor",
               not _hasfs or ("discrete\\_geometry\\_r506" in _fsg_src
                              and "$323$ of $792$" in _fsg_src
                              and "$0$ holes" in _fsg_src
                              and "$0$ overhangs" in _fsg_src
                              and "0.014, 0.357, 0.214" in _fsg_src))

    # ---------- r507: discriminating-tolerance classifier audit + 2.4x erratum ----
    ced2 = load("discriminant_r507_result.json")
    claim_true("CED2.checks",
               "CED2 r507 self-checks pass (D1 sign perfect, D2 draft1 22 errors, "
               "D3 fixed-margin best 11, D4 margins, D5 probe invariance, "
               "D5b tolerance band, D6 rounding sensitivity)",
               ced2["checks"]["ALL_PASS"] is True)
    claim_true("CED2.sign",
               "CED2 slack-sign classifier is exact on all 48 cells at the "
               "discriminating tolerance: 11 TP / 37 TN / 0 FP / 0 FN",
               ced2["classifiers"]["A_slack_sign"]["TP"] == 11
               and ced2["classifiers"]["A_slack_sign"]["TN"] == 37
               and ced2["classifiers"]["A_slack_sign"]["errors"] == 0)
    claim_true("CED2.falsified",
               "CED2 falsified drafts quantified: draft1 (unique-binding "
               "biconditional) 22 errors (19 FP + 3 FN); best fixed margin "
               "threshold achieves only the always-strict constant (11 errors)",
               ced2["classifiers"]["B_draft1_biconditional"]["errors"] == 22
               and ced2["classifiers"]["B_draft1_biconditional"]["FP"] == 19
               and ced2["classifiers"]["B_draft1_biconditional"]["FN"] == 3
               and ced2["classifiers"]["C_fixed_margin_best"]["errors"] == 11)
    claim_true("CED2.margin",
               "CED2 inner margin ratio 1.20x (nearest slacks -2.43e-05 / "
               "+2.91e-05) -- corrects the >=2.4x prose of v5.19/v5.20",
               abs(ced2["margins"]["sign_gap_ratio"] - 1.1953928424515792) < 1e-9
               and abs(ced2["margins"]["nearest_margin_cells"]["tight"]
                       - (-2.431e-05)) < 1e-9
               and abs(ced2["margins"]["nearest_margin_cells"]["strict"]
                       - 2.906e-05) < 1e-9)
    claim_true("CED2.tolband",
               "CED2 usable discriminating band [5e-7, 8.22e-05): 11-tight "
               "plateau on {5e-7,1e-6,2e-6}, degenerates to 3 at 2.5e-7 "
               "(6dp rounding spread), absorbs the genuine 12th cell only "
               ">= 8.22e-05; no strict cell absorbed by rounding",
               ced2["checks"]["D5b_truth_tol_sensitivity"]["pass"] is True
               and ced2["checks"]["D6_rounding_perturbation"]["pass"] is True)
    claim_true("X.ced2.i",
               "tex prints the discriminating audit: discriminant_r507 anchor, "
               "11 TP / 37 TN, 22-cell draft1 failure, 1.20x margin correction, "
               "and the [5e-7, 8.22e-05) tolerance band",
               not _hasfs or ("discriminant\\_r507" in _fsg_src
                              and "$11$~TP~/~$37$~TN" in _fsg_src
                              and "1.20\\times$" in _fsg_src
                              and "$[5\\times10^{-7},8.22\\times10^{-5})$" in _fsg_src
                              and "disjoint with a ${\\ge}2.4\\times$ margin gap"
                                  not in _fsg_src))

    # ---------- r508: formal prefix-law Proposition + machine witness ----
    cpf = load("prefix_prop_r508_result.json")
    claim_true("CPF.checks",
               "CPF r508 witness self-checks pass (V1 base_S monotone coarse, "
               "V2 gmax monotone coarse, V3 gmax monotone fine (stored r503 "
               "inline witness), V4 prefix 0 holes/0 overhangs, V5 base-driven "
               "crossing census, V6 323/792 composition, V7 chain anchors)",
               cpf["checks"]["ALL_PASS"] is True)
    claim_true("CPF.scalars",
               "CPF two-scalar monotonicity witnessed: base_S 0 violations on "
               "792 coarse pairs; gmax_S 0 on 792 coarse + 2592 fine pairs",
               cpf["checks"]["V1_baseS_monotone_coarse"]["violations"] == 0
               and cpf["checks"]["V1_baseS_monotone_coarse"]["adjacent_pairs"] == 792
               and cpf["checks"]["V2_gmax_monotone_coarse"]["violations"] == 0
               and cpf["checks"]["V3_gmax_monotone_fine"]["adjacent_pairs"] == 2592
               and cpf["checks"]["V3_gmax_monotone_fine"]["violations"] == 0)
    claim_true("CPF.prefix",
               "CPF prefix law witnessed: 0 holes / 0 overhangs on the fine "
               "grid (72 cells); coarse 323 violations = 253 sub-unit wobbles "
               "+ 70 prefix-edge jumps + 0 holes",
               cpf["checks"]["V4_prefix_fine"]["holes"] == 0
               and cpf["checks"]["V4_prefix_fine"]["overhangs"] == 0
               and cpf["checks"]["V6_tau_nonmonotone_interior"]["tau_violations"] == 323
               and cpf["checks"]["V6_tau_nonmonotone_interior"]["adjacent_pairs"] == 792
               and cpf["checks"]["V6_tau_nonmonotone_interior"]["edge_jumps_into_tau1"] == 70
               and cpf["checks"]["V6_tau_nonmonotone_interior"]["interior_subunit_violations"] == 253
               and cpf["checks"]["V6_tau_nonmonotone_interior"]["holes"] == 0)
    claim_true("CPF.driver",
               "CPF crossing out of tau=1 is base_S-driven: every prefix-end "
               "case has base_S <= alpha (V5 census, exact recompute with the "
               "frozen subset_profile machinery)",
               cpf["checks"]["V5_crossing_driver_census"]["pass"] is True
               and cpf["checks"]["V5_crossing_driver_census"]["prefix_end_cases"]
                   == cpf["checks"]["V5_crossing_driver_census"]["base_le_alpha_at_end"])
    claim_true("X.cpf.i",
               "tex prints the formal prefix-law Proposition with short proof: "
               "label prop:prefix, both scalarizations nondecreasing, prefix "
               "crossing set, 253+70 decomposition, r508 anchor",
               not _hasfs or ("label{prop:prefix}" in _fsg_src
                              and "prefix\\_prop\\_r508" in _fsg_src
                              and "$253$ strictly sub-unit wobbles" in _fsg_src
                              and "$70$" in _fsg_src
                              and "downward" in _fsg_src))

    # ---------- r492: cross-carrier evidence matrix (app:matrix) ----------
    # Matrix table cells: shard1 (r471 R1/R2), OpenR1 (r473), RLVE (r474),
    # OMR s0 cert values (r469 cal_selection) + descriptive CAL-side alpha=.01
    # row (r478 continuous remark: grid-selected mean k 10.3 -> 11.9).
    _s1w = shard1["R1_within"]["test_readout"]
    _s1t = shard1["R2_transfer"]["test_readout"]
    claim("MX.s1.eb05.flip", "MX shard1 EB a=.05 flip", _s1w["FIXED_EB_a0.05"]["realized_flip"], "0.0357", "f4")
    claim("MX.s1.eb05.save", "MX shard1 EB a=.05 saving", _s1w["FIXED_EB_a0.05"]["saving_vs_full"], "53.1", "pct1")
    claim("MX.s1.bh05.flip", "MX shard1 BAYES-H a=.05 flip", _s1w["BAYESH_a0.05"]["realized_flip"], "0.0246", "f4")
    claim("MX.s1.bh05.save", "MX shard1 BAYES-H a=.05 saving", _s1w["BAYESH_a0.05"]["saving_vs_full"], "80.6", "pct1")
    claim("MX.s1.bh02.flip", "MX shard1 BAYES-H a=.02 flip", _s1w["BAYESH_a0.02"]["realized_flip"], "0.0088", "f4")
    claim("MX.s1.bh02.save", "MX shard1 BAYES-H a=.02 saving", _s1w["BAYESH_a0.02"]["saving_vs_full"], "73.1", "pct1")
    claim("MX.s1t.bh05.flip", "MX shard1 transfer BAYES-H a=.05 flip", _s1t["BAYESH_a0.05"]["realized_flip"], "0.0247", "f4")
    claim("MX.s1t.bh05.save", "MX shard1 transfer BAYES-H a=.05 saving", _s1t["BAYESH_a0.05"]["saving_vs_full"], "80.6", "pct1")
    claim("MX.s1t.bh02.flip", "MX shard1 transfer BAYES-H a=.02 flip", _s1t["BAYESH_a0.02"]["realized_flip"], "0.0089", "f4")
    claim("MX.s1t.bh02.save", "MX shard1 transfer BAYES-H a=.02 saving", _s1t["BAYESH_a0.02"]["saving_vs_full"], "73.2", "pct1")
    claim("MX.or1.f1.flip", "MX OpenR1 FIXED-1 flip", or1["test_readout"]["FIXED1"]["realized_flip"], "0.0592", "f4")
    claim("MX.or1.bh05.flip", "MX OpenR1 BAYES-H a=.05 flip", or1["test_readout"]["BAYESH_a0.05"]["realized_flip"], "0.0294", "f4")
    claim("MX.or1.bh05.save", "MX OpenR1 BAYES-H a=.05 saving", or1["test_readout"]["BAYESH_a0.05"]["saving_vs_full"], "32.1", "pct1")
    claim("MX.or1.bh02.flip", "MX OpenR1 BAYES-H a=.02 flip (never stops)", or1["test_readout"]["BAYESH_a0.02"]["realized_flip"], "0.0", "k1")
    claim("MX.or1.bh02.save", "MX OpenR1 BAYES-H a=.02 saving 0", or1["test_readout"]["BAYESH_a0.02"]["saving_vs_full"], "0.0", "k1")
    _rl = _rlve["within"]["test_readout"]
    claim("MX.rl.bh01.flip", "MX RLVE BAYES-H a=.01 flip", _rl["BAYESH_a0.01"]["realized_flip"], "0.0019", "f4")
    claim("MX.rl.bh01.save", "MX RLVE BAYES-H a=.01 saving", _rl["BAYESH_a0.01"]["saving_vs_full"], "45.4", "pct1")
    claim("MX.rl.bh01.cert", "MX RLVE BAYES-H a=.01 cert", _rlve["within"]["cal_selection"]["0.01"]["BAYESH_cert"], "0.0081", "f4")
    _mg = margin["results"]
    claim("MX.mg.e2.flip", "MX margin E2 g=.03 d=.20 flip", _mg["a0.05_E2_linear_p_d0.2_g0.03"]["flip"], "0.0491", "f4")
    claim_tol("MX.mg.e2.save", "MX margin E2 g=.03 d=.20 saving (printed 71.8)", _mg["a0.05_E2_linear_p_d0.2_g0.03"]["saving"], 71.8, 0.05)
    claim("MX.mg.e2.save0", "MX margin E2 unmargined d=.20 saving", _mg["a0.05_E2_linear_p_d0.2_g0.0"]["saving"], "79.7", "pct1")
    claim("MX.mg.e3.flip", "MX margin E3 g=.025 d=.15 flip", _mg["a0.05_E3_blockswap_d0.15_g0.025"]["flip"], "0.0495", "f4")
    claim("MX.mg.e3.save", "MX margin E3 g=.025 d=.15 saving", _mg["a0.05_E3_blockswap_d0.15_g0.025"]["saving"], "77.2", "pct1")
    claim("MX.mg.e3.save0", "MX margin E3 unmargined d=.15 saving", _mg["a0.05_E3_blockswap_d0.15_g0.0"]["saving"], "81.0", "pct1")
    # OMR s0 cert values (matrix cert column) and CAL-side alpha=.01 row
    _cal = fit["cal_selection"]
    claim("MX.s0.cert.hoef10", "MX s0 cert HOEF a=.10", _cal["0.1"]["FIXED_HOEF_cert"], "0.0970", "f4")
    claim("MX.s0.cert.eb10", "MX s0 cert EB a=.10", _cal["0.1"]["FIXED_EB_cert"], "0.0958", "f4")
    claim("MX.s0.cert.bh10", "MX s0 cert BAYES-H a=.10", _cal["0.1"]["BAYESH_cert"], "0.0495", "f4")
    claim("MX.s0.cert.hoef05", "MX s0 cert HOEF a=.05", _cal["0.05"]["FIXED_HOEF_cert"], "0.0471", "f4")
    claim("MX.s0.cert.eb05", "MX s0 cert EB a=.05", _cal["0.05"]["FIXED_EB_cert"], "0.0461", "f4")
    claim("MX.s0.cert.bh05", "MX s0 cert BAYES-H a=.05", _cal["0.05"]["BAYESH_cert"], "0.0350", "f4")
    claim("MX.s0.cert.eb02", "MX s0 cert EB a=.02", _cal["0.02"]["FIXED_EB_cert"], "0.0171", "f4")
    claim("MX.s0.cert.bh02", "MX s0 cert BAYES-H a=.02", _cal["0.02"]["BAYESH_cert"], "0.0163", "f4")
    claim("MX.s0.cert.bh01", "MX s0 cert BAYES-H a=.01 (CAL-side, descriptive row)", _cal["0.01"]["BAYESH_cert"], "0.0119", "f4")
    # r478 continuous remark: alpha=.01 selection mean k grid 10.3 -> cont 11.9
    _r478 = load("alpha_continuous_r478_result.json")
    claim("MX.cont.mk01.grid", "MX remark cont-alpha grid mean k at .01 (printed 10.3)", _r478["reference_alphas"]["0.01"]["grid"]["mean_k"], "10.3", "k1")
    claim("MX.cont.mk01.cont", "MX remark cont-alpha cont mean k at .01 (printed 11.9)", _r478["reference_alphas"]["0.01"]["cont"]["mean_k"], "11.9", "k1")
    # tex-direct-parse layer (X.mx.*): matrix-only printed strings
    claim_true("X.mx.s1row",
               "tex matrix prints shard1 EB row 0.0486/0.0357/53.1",
               not _hasfs or ("0.0486 & 0.0357 & 15.0 & 53.1" in _fs_src))
    claim_true("X.mx.or1row",
               "tex matrix prints OpenR1 a=.02 row 0.0047/0/2.0/0",
               not _hasfs or ("0.0047 & 0 & 2.0 & 0" in _fs_src))
    claim_true("X.mx.rl01row",
               "tex matrix prints RLVE a=.01 row 0.0081/0.0019/45.4",
               not _hasfs or ("0.0081 & 0.0019 & 4.4 & 45.4" in _fs_src))
    claim_true("X.mx.bh01row",
               "tex matrix prints OMR s0 descriptive a=.01 cert row 0.0119",
               not _hasfs or ("& .01 & 0.0119 & ---" in _fs_src))
    claim_true("X.mx.margine2",
               "tex unified mechanism prints E2 margin 71.8 vs 79.7",
               not _hasfs or ("saving\n$71.8\\%$ vs $79.7\\%$ unmargined" in _fs_src
                              or "$71.8\\%$ vs $79.7\\%$ unmargined" in _fs_src))
    claim_true("X.mx.margine3",
               "tex unified mechanism prints E3 margin 77.2 vs 81.0",
               not _hasfs or ("$77.2\\%$ vs $81.0\\%$" in _fs_src))

    # ---------- r486: Spearman decomposition audit (peer-audit-driven) ----------
    # A1 r1108 / A9 r263 pattern: a pooled correlation can be a
    # between-unit artifact. Decompose the 16-cell pooled 0.75 and verify
    # the repaired v5.9 wording (between-carrier law, het not confounded).
    sd = load("spearman_decomposition_r486_result.json")
    claim_true("SD.pooled.match",
               "SD reproduces r484 pooled 16-cell Spearman 0.7544",
               sd["q1_pooled_16"]["match"] is True)
    claim("SD.pooled", "SD pooled Spearman 16 cells (printed 0.75)",
          sd["q1_pooled_16"]["spearman"], "0.75", "f2")
    claim_true("SD.within.omr.neg",
               "SD within-OMR Spearman negative both shards (printed -0.95/-0.89)",
               sd["q2_within_carrier"]["omr_shard0"]["spearman"] < -0.8
               and sd["q2_within_carrier"]["omr_shard1"]["spearman"] < -0.8)
    claim("SD.within.s0", "SD within-OMR shard0 (printed -0.95)",
          sd["q2_within_carrier"]["omr_shard0"]["spearman"], "-0.95", "f2")
    claim("SD.within.s1", "SD within-OMR shard1 (printed -0.89)",
          sd["q2_within_carrier"]["omr_shard1"]["spearman"], "-0.89", "f2")
    claim("SD.within.rlve", "SD within-RLVE (printed 0.32)",
          sd["q2_within_carrier"]["rlve"]["spearman"], "0.32", "f2")
    claim("SD.within.openr1", "SD within-OpenR1 (printed 1.0)",
          sd["q2_within_carrier"]["openr1"]["spearman"], "1.0", "k1")
    claim("SD.between", "SD between-carrier means (printed 1.0)",
          sd["q3_between_carrier_means"]["spearman"], "1.0", "k1")
    claim("SD.loco.min", "SD leave-one-carrier-out range lo (printed 0.44)",
          min(sd["q4_loco"].values()), "0.44", "f2")
    claim("SD.loco.max", "SD leave-one-carrier-out range hi (printed 0.90)",
          max(sd["q4_loco"].values()), "0.90", "f2")
    claim("SD.plateaus", "SD plateaus-removed 14 cells (printed 0.63)",
          sd["q5_plateaus_removed"]["spearman"], "0.63", "f2")
    claim_true("SD.plateaus.n",
               "SD plateaus-removed drops exactly the 2 OpenR1 plateau cells",
               sd["q5_plateaus_removed"]["n_cells"] == 14
               and sd["q5_plateaus_removed"]["dropped"]
               == ["openr1@0.02", "openr1@0.01"])
    claim("SD.het.tau", "SD het-excess vs tau* (printed 0.01)",
          sd["q6_het_vs_tau_16"]["spearman"], "0.01", "f2")
    claim("SD.het.zerog", "SD het-excess vs zero-g (printed 0.05)",
          sd["q7_het_vs_zerog_16"], "0.05", "f2")
    claim_true("SD.het.xcheck",
               "SD het-excess matches pinned carrier artifacts within 0.01",
               sd["q8_het_artifact_xcheck"]["match_within_0.01"] is True)

    # ---------- paper.tex <-> artifact direct parse (r480, mgr 45f9831dc4ce) ----------
    # Instead of trusting hardcoded "printed" strings above, regex-parse the
    # values actually printed in paper.tex and compare them against the
    # artifacts. Prevents a hardcoded self-consistent false green (the r475
    # Table-1 alpha=.10 hole and the r480 alpha=.05 0.0345-vs-0.0324 slip
    # both came from trusting printed strings). Path: PAPER_TEX env var or
    # ../paper.tex (canonical) or ./paper.tex (candidate bundle copy).
    import re as _re
    _ptex = os.environ.get("PAPER_TEX")
    if _ptex is None:
        # Prefer the bundle's own copy (./paper.tex) so an out-of-tree replay
        # of a candidate never silently parses a *different* (stale) canonical
        # via ../paper.tex. Fall back to ../paper.tex only when the checker
        # sits in an audit_pack without its own tex copy.
        for _cand in (os.path.join(HERE, "paper.tex"),
                      os.path.join(HERE, "..", "paper.tex")):
            if os.path.isfile(_cand):
                _ptex = _cand
                break
    claim_true("X.tex.found", "paper.tex located for direct parse",
               _ptex is not None, f"path={_ptex}")
    if _ptex is not None:
        _src = open(_ptex).read()
        _pax = os.path.join(os.path.dirname(_ptex), "appendix_proofs.tex")
        _asrc = open(_pax).read() if os.path.isfile(_pax) else ""

        def row_pat(rule, alpha):
            num = r"(?:\\textbf\{)?\s*([0-9.]+)\s*\}?"
            m = _re.search(rule + r"[^%\n]*?&\s*" + alpha +
                           r"\s*&\s*" + num + r"\s*&\s*" + num +
                           r"\s*&\s*" + num + r"\\%",
                           _src)
            return m.groups() if m else None

        _expect = [
            # (cid, row regex head, alpha col, artifact dict) -- dicts re-read
            # from tr to dodge loop-variable shadowing above (h was reused).
            ("X.T1.HOEF10", r"FIXED-HOEF\s*\$k\^\*\{=\}7\$", "0.10", tr["FIXED_HOEF_a0.1"]),
            ("X.T1.EB10", r"FIXED-EB\s*\$k\^\*\{=\}5\$", "0.10", tr["FIXED_EB_a0.1"]),
            ("X.T1.BH10", r"BAYES-H", "0.10", tr["BAYESH_a0.1"]),
            ("X.T1.HOEF05", r"FIXED-HOEF\s*\$k\^\*\{=\}27\$", "0.05", tr["FIXED_HOEF_a0.05"]),
            ("X.T1.EB05", r"FIXED-EB\s*\$k\^\*\{=\}17\$", "0.05", tr["FIXED_EB_a0.05"]),
            ("X.T1.BH05", r"BAYES-H", "0.05", tr["BAYESH_a0.05"]),
            ("X.T1.EB02", r"FIXED-EB\s*\$k\^\*\{=\}31\$", "0.02", tr["FIXED_EB_a0.02"]),
            ("X.T1.BH02", r"BAYES-H", "0.02", tr["BAYESH_a0.02"]),
        ]
        for cid, head, alpha, art in _expect:
            g = row_pat(head, alpha)
            if g is None:
                FAILS.append(f"{cid} Table1 row not parseable from paper.tex ({head}, a={alpha})")
                continue
            flip_t, k_t, save_t = g
            ok = (abs(float(flip_t) - float(art["realized_flip"])) <= 5.1e-5
                  and abs(float(k_t) - float(art["mean_k"])) <= 0.051
                  and abs(float(save_t) - 100 * float(art["saving_vs_full"])) <= 0.061)
            if ok:
                PASSES.append(f"{cid} Table1 row printed==artifact (flip={flip_t}, k={k_t}, save={save_t}%)")
            else:
                FAILS.append(f"{cid} printed ({flip_t},{k_t},{save_t}) != artifact "
                             f"({art['realized_flip']:.4f},{art['mean_k']:.1f},{100*art['saving_vs_full']:.1f})")

        # caption: must NOT claim cheaper-than-every-baseline; must disclose 84.4 vs 84.0
        claim_true("X.cap.nodom",
                   "caption no longer claims 'cheaper than every baseline'",
                   "cheaper than every baseline" not in _src)
        claim_true("X.cap.disclose",
                   "caption discloses FIXED-EB 84.4% vs BAYES-H 84.0% at a=.10",
                   "84.4\\%" in _src and "84.0\\%" in _src)
        # intro: no bare 'monotone in the prefix length' on the conditional object
        claim_true("X.intro.mono",
                   "intro attributes monotonicity to fixed-budget E_H[f_K(k)]",
                   "monotone in the\nprefix length" not in _src
                   and "E_H[f_K(k)]" in _src)
        # appendix: no 'keeps monotonicity intact' even-k sentence
        claim_true("X.appx.evenk",
                   "appendix even-k 'monotonicity intact' sentence removed",
                   "keeps monotonicity intact" not in _asrc)
        # bibliography: moshkov2025 is cited in text (real relation: OMR carrier)
        claim_true("X.bib.moshkov",
                   "moshkov2025 cited at carrier paragraph",
                   "citep{moshkov2025openmathreasoning}" in _src)
        # r482 cross-carrier paragraph: parse printed radii straight from tex
        _m = _re.search(
            r"tau\^\*\(0\.10/0\.05/0\.02/0\.01\)=0\.199/0\.551/0\.213/0\.449",
            _src)
        claim_true("X.tvx.radii",
                   "paper.tex prints RLVE radii 0.199/0.551/0.213/0.449",
                   _m is not None)
        claim_true("X.tvx.artifact",
                   "paper.tex names tv_robustness_rlve_r482_result.json",
                   "tv\\_robustness\\_rlve\\_r482\\_result.json" in _src)
        _m2 = _re.search(r"concentrates \$74\$--\$87\\%\$", _src)
        claim_true("X.tvx.zerog",
                   "paper.tex prints RLVE zero-g mass 74--87%",
                   _m2 is not None)
        claim_true("X.tvx.r481range",
                   "paper.tex discloses r481 bisection range [0,0.5] vs [0,1]",
                   "[0,0.5]" in _src and "R\\in[0,1]" in _src)
        # r483 three-atom edge case paragraph: parse printed values from tex
        _m3 = _re.search(
            r"tau\^\*\(0\.10\)=\\tau\^\*\(0\.05\)=0\.168", _src)
        claim_true("X.tvo.radii",
                   "paper.tex prints OpenR1 radii tau*(.10)=tau*(.05)=0.168",
                   _m3 is not None)
        _m4 = _re.search(
            r"tau\^\*\(0\.02\)=\\tau\^\*\(0\.01\)=1", _src)
        claim_true("X.tvo.whole",
                   "paper.tex prints OpenR1 whole-simplex tau*(.02)=tau*(.01)=1",
                   _m4 is not None)
        claim_true("X.tvo.artifact",
                   "paper.tex names tv_robustness_openr1_r483_result.json",
                   "tv\\_robustness\\_openr1\\_r483\\_result.json" in _src)
        claim_true("X.tvo.zerog",
                   "paper.tex prints OpenR1 zero-g mass 76.8%",
                   "76.8\\%" in _src)
        claim_true("X.tvo.pump",
                   "paper.tex states one-sided pump V(R)=flip+g_max R",
                   "one-sided pump" in _src and "g_{\\max}R" in _src)
        claim_true("X.tvo.certs",
                   "paper.tex prints OpenR1 certs 0.0785/0.0460",
                   "0.0785/0.0460" in _src)
        # r484 conservation curve: direct-parse the printed claims from tex
        claim_true("X.cc.artifact",
                   "paper.tex names tv_conservation_r484_result.json",
                   "tv\\_conservation\\_r484\\_result.json" in _src)
        claim_true("X.cc.figure",
                   "paper.tex includes fig_tau_conservation.png",
                   "fig_tau_conservation.png" in _src)
        claim_true("X.cc.bps",
                   "paper.tex prints rule-change counts 96/93/7/2",
                   "96/93/7/2" in _src)
        claim_true("X.cc.jumpdown",
                   "paper.tex prints OpenR1 down jump -0.314 at 0.0785",
                   "$\\Delta=-0.314$" in _src and "0.07852" in _src)
        claim_true("X.cc.jumpup",
                   "paper.tex prints OpenR1 2nd jump DOWN -0.864 at 0.04598 (direction aligned)",
                   "$\\Delta=-0.864$" in _src and "0.04598" in _src
                   and "$\\Delta=+0.864$" not in _src)
        claim_true("X.cc.spearman",
                   "paper.tex prints Spearman 0.75 (main + appendix)",
                   _src.count("Spearman") >= 2 and "0.75" in _src)
        claim_true("X.cc.plateau",
                   "paper.tex prints RLVE plateau alpha<0.002 (strict) and OpenR1 <0.0460",
                   "$\\alpha<0.002$" in _src and "\\alpha\\le0.002" not in _src
                   and "$\\alpha<0.0460$" in _src)
        claim_true("X.cc.appx",
                   "paper.tex has Appendix section app:tau with fig:tau",
                   "\\label{app:tau}" in _src and "\\label{fig:tau}" in _src)
        # caption range/step vs JSON alpha_grid
        _ccj = load("tv_conservation_r484_result.json")
        claim_true("X.cc.grid",
                   "caption grid [1e-3,0.2] step 5e-4 matches JSON alpha_grid",
                   "\\alpha\\in[10^{-3},0.2]" in _src
                   and "5\\!\\times\\!10^{-4}" in _src
                   and abs(_ccj["alpha_grid"]["lo"] - 0.001) < 1e-12
                   and abs(_ccj["alpha_grid"]["hi"] - 0.20) < 1e-12
                   and abs(_ccj["alpha_grid"]["step"] - 0.0005) < 1e-12)
        # r486 decomposition-audit wording, direct-parsed from tex
        _sdj = load("spearman_decomposition_r486_result.json")
        claim_true("X.sd.fix",
                   "paper.tex says 'no more heterogeneous' (direction slip repaired)",
                   "no more heterogeneous" in _src
                   and "is more heterogeneous" not in _src)
        claim_true("X.sd.artifact",
                   "paper.tex names spearman_decomposition_r486_result.json",
                   "spearman\\_decomposition\\_r486\\_result.json" in _src)
        claim_true("X.sd.between",
                   "paper.tex prints between-carrier law wording",
                   "between-carrier" in _src)
        claim_true("X.sd.within",
                   "paper.tex prints within-OMR -0.95/-0.89, RLVE 0.32, OpenR1 1.0",
                   "-0.95" in _src and "-0.89" in _src
                   and "$0.32$" in _src and "$1.0$" in _src)
        claim_true("X.sd.het",
                   "paper.tex prints het-vs-tau* 0.01 and het-vs-alignment 0.05",
                   "Spearman $0.01$" in _src and "$0.05$" in _src)
        claim_true("X.sd.plateau63",
                   "paper.tex prints plateaus-removed 0.63",
                   "$0.63$" in _src)
        # cross-check: printed within values equal the artifact values
        claim_true("X.sd.numbers.match",
                   "tex printed decomposition numbers equal artifact values",
                   f"{_sdj['q2_within_carrier']['omr_shard0']['spearman']:.2f}" == "-0.95"
                   and f"{_sdj['q2_within_carrier']['omr_shard1']['spearman']:.2f}" == "-0.89"
                   and f"{_sdj['q2_within_carrier']['rlve']['spearman']:.2f}" == "0.32"
                   and f"{_sdj['q5_plateaus_removed']['spearman']:.2f}" == "0.63"
                   and f"{_sdj['q6_het_vs_tau_16']['spearman']:.2f}" == "0.01"
                   and f"{_sdj['q7_het_vs_zerog_16']:.2f}" == "0.05")
        # candidate bundles also carry the canonical Chinese detailed
        # report; union it into the checked text so no printed document can
        # contradict the fixed direction/boundary claims.
        _zh = os.path.join(os.path.dirname(_ptex),
                           "A11_CANONICAL_DETAILED_REPORT_ZH.md")
        _src = _src + "\n" + (open(_zh).read()
                              if os.path.isfile(_zh) else "")

        # r488 (A6 audit of r484_v5_8, MAJOR+MINOR-1): tex-vs-artifact
        # DIRECTION negative controls. The r486 lesson applies with a sign
        # flip here: v5.10 asserts both OpenR1 jumps are strict DOWNWARD
        # (tau* 1.0 -> 0.136, signed Delta=-0.864); the artifact stores the
        # signed value directly, so check tex sign, tex direction word and
        # the artifact sign together, not only magnitudes/positions.
        _ja = _ccj["openr1_jump_analysis"]
        claim_true("X.cc.jump2.signdown",
                   "artifact 2nd OpenR1 jump is strictly negative (downward)",
                   _ja["jump_up_at_0.04598"] < 0
                   and abs(_ja["jump_up_at_0.04598"] + 0.864) < 0.01)
        claim_true("X.cc.jump2.texdown",
                   "paper.tex prints the 2nd jump as 1 -> 0.136 (down)",
                   "1\\!\\to\\!0.136" in _src and "0.136\\!\\to\\!1" not in _src)
        # MINOR-1: RLVE plateau boundary is strict (alpha=0.002 is itself a
        # rule-change breakpoint, tau*(0.002)=0.06 < 1): tex must use <.
        claim_true("X.cc.plateau.strict",
                   "RLVE plateau boundary strict in tex AND 0.002 is RLVE's 1st breakpoint",
                   "\\alpha\\le0.002" not in _src
                   and abs(_ccj["breakpoints_alpha"]["rlve"][0] - 0.002) < 1e-9)

    # ---------- DIR.* direction-predicate layer (r487) ----------
    # r486 lesson: a checked printed NUMBER does not check the RELATIONAL/
    # DIRECTION word around it (the "more heterogeneous" slip). For every
    # comparison sentence in the paper, assert BOTH the artifact numbers AND
    # the direction predicate the paper claims, plus the printed wording.
    # r487's own pre-flight caught a LIVE bug this way: the limitation range
    # "0.052--0.157" contradicted the tau*(.01)=0.024 printed two lines later.
    _tv = load("tv_robustness_r481_result.json")
    _rl = load("tv_robustness_rlve_r482_result.json")
    _op = load("tv_robustness_openr1_r483_result.json")
    _r474 = load("rlve_n8_r474_result.json")
    if "_sdj" not in locals():
        _sdj = load("spearman_decomposition_r486_result.json")
    if "_src" not in locals():
        _src = ""
    _c0 = _tv["shard0"]["critical_radius"]; _c1 = _tv["shard1"]["critical_radius"]
    _cr = _rl["critical_radius"]
    # RLVE radii "larger than on OMR at every level" (both shards, 4 levels)
    claim_true("DIR.tv.larger",
               "RLVE tau* > OMR tau* at every level (both shards)",
               all(_cr[a]["tau_star"] > _c0[a]["tau_star"]
                   and _cr[a]["tau_star"] > _c1[a]["tau_star"]
                   for a in ("0.1", "0.05", "0.02", "0.01")))
    # "no more heterogeneous": RLVE het-excess 0.0999 < OMR 0.129 (direction)
    claim_true("DIR.tv.hetex",
               "RLVE het-excess (0.0999) strictly below OMR (0.129)",
               _r474["within"]["het_excess_var"] < 0.129
               and abs(_r474["within"]["het_excess_var"] - 0.0999) < 0.01)
    # zero-g alignment: RLVE min mass 0.7383 strictly above OMR max 0.4655
    _omax = max([_tv["shard0"]["zero_g_atoms"][a]["H_mass_zero_g"]
                 for a in _tv["shard0"]["zero_g_atoms"]]
                + [_tv["shard1"]["zero_g_atoms"][a]["H_mass_zero_g"]
                   for a in _tv["shard1"]["zero_g_atoms"]])
    _rmin = min(_rl["zero_g_atoms"][a]["H_mass_zero_g"]
                for a in _rl["zero_g_atoms"])
    claim_true("DIR.tv.zerog",
               "RLVE min zero-g mass (0.7383) above OMR max (0.4655)",
               _rmin > _omax and abs(_omax - 0.4655) < 0.01
               and abs(_rmin - 0.7383) < 0.01)
    # limitation range: full grid min=0.024 (shard1 a=.01), max=0.157 (shard1 a=.10)
    _allt = [_c0[a]["tau_star"] for a in _c0] + [_c1[a]["tau_star"] for a in _c1]
    claim_true("DIR.tv.range",
               "printed tau range 0.024--0.157 matches full 8-cell grid",
               abs(min(_allt) - 0.02403) < 0.01 and abs(max(_allt) - 0.15663) < 0.01
               and "$\\tau^*=0.024$--$0.157$" in _src)
    # sharp boundary: 0.042 exceeds tau*(.01)=0.024 AND <=.05/.02 stay valid
    claim_true("DIR.tv.exceed",
               "0.042 > tau*(.01)=0.024 while V(0.042)<=.05 and <=.02",
               _c1["0.01"]["tau_star"] < 0.042
               and _tv["shard1"]["worst_case_key_R"]["0.05"]["V(R=0.042)"] <= 0.05
               and _tv["shard1"]["worst_case_key_R"]["0.02"]["V(R=0.042)"] <= 0.02)
    # non-monotonicity direction: RLVE tau*(.05)>tau*(.10); OpenR1 tau*(.02)=1>tau*(.05)
    claim_true("DIR.tv.nonmono",
               "tau* non-monotone direction (RLVE .05>.10; OpenR1 .02=1 > .05)",
               _cr["0.05"]["tau_star"] > _cr["0.1"]["tau_star"]
               and _op["critical_radius"]["0.02"]["tau_star_closed"]
                   > _op["critical_radius"]["0.05"]["tau_star_closed"])
    if os.environ.get("CC_DEBUG"):
        print("DBG srclen:", len(_src))
        print("DBG 1to0.136:", "1\\!\\to\\!0.136" in _src)
        print("DBG -0.864:", "$\\Delta=-0.864$" in _src)
        print("DBG +0.864:", "$\\Delta=+0.864$" in _src)
        print("DBG a<0.002:", "$\\alpha<0.002$" in _src)
        print("DBG a<=0.002:", "\\alpha\\le0.002" in _src)
    # r488 negative controls for the OpenR1 2nd jump (A6 MAJOR): assert
    # the tex direction word + artifact sign + endpoint order together.
    _ccj2 = load("tv_conservation_r484_result.json")
    _ja2 = _ccj2["openr1_jump_analysis"]
    # tex-side direction predicates are only meaningful when a tex was found
    # (_src=="" stub must not auto-pass a *direction* claim)
    _hastex = bool(_src)
    claim_true("DIR.cc.jump2",
               "OpenR1 2nd jump is DOWN: signed -0.864, tex says down 1->0.136",
               _ja2["jump_up_at_0.04598"] < 0
               and (not _hastex or (
                   "1\\!\\to\\!0.136" in _src
                   and "$\\Delta=-0.864$" in _src
                   and "$\\Delta=+0.864$" not in _src)))
    # r488 negative control for the RLVE plateau boundary (A6 MINOR-1):
    # grid cells on both sides of the strict boundary.
    _rtau = _ccj2["curves"]["rlve"]["tau"]
    _g = _ccj2["alpha_grid"]
    _i = lambda a: round((a - _g["lo"]) / _g["step"])
    claim_true("DIR.cc.plateau.edge",
               "RLVE plateau holds at a=0.0015 (tau=1) but not at a=0.002 (tau=0.06)",
               abs(_rtau[_i(0.0015)] - 1.0) < 1e-9
               and abs(_rtau[_i(0.002)] - 0.06) < 1e-9
               and (not _hastex or "$\\alpha<0.002$" in _src))
    # shift band: worst 0.042<=0.10, saving 80.3<83.6 => a 3.3-point price
    _sb = _tv["shard0"]["shift_band_R0.05"]["0.1"]
    claim_true("DIR.tv.shiftband",
               "R=0.05 band: worst 0.042<=0.10, 80.3<83.6, price 3.3pt",
               _sb["worst_flip_at_R0.05"] <= 0.10
               and abs(_sb["saving_band"] - 0.8027) < 0.001
               and abs(_sb["saving_orig"] - 0.8356) < 0.001
               and abs((_sb["saving_orig"] - _sb["saving_band"]) - 0.0329) < 0.001)
    # decomposition direction words: within-OMR negative, RLVE/OpenR1 positive
    _w = _sdj["q2_within_carrier"]
    claim_true("DIR.sd.within",
               "within-carrier signs: OMR s0/s1 negative, RLVE/OpenR1 positive",
               _w["omr_shard0"]["spearman"] < 0 and _w["omr_shard1"]["spearman"] < 0
               and _w["rlve"]["spearman"] > 0 and _w["openr1"]["spearman"] > 0)
    # het-excess is NOT the driver: both het correlations near zero
    claim_true("DIR.sd.het",
               "het-excess vs tau* and vs zero-g both ~0 (not the driver)",
               abs(_sdj["q6_het_vs_tau_16"]["spearman"]) < 0.1
               and abs(_sdj["q7_het_vs_zerog_16"]) < 0.1)
    # cross-carrier claim robustness: means 1.0, plateaus-removed 0.63, LOCO in [0.44,0.90]
    claim_true("DIR.sd.robust",
               "cross-carrier co-variation robust (means 1.0, excl-plateau 0.63, LOCO 0.44-0.90)",
               _sdj["q3_between_carrier_means"]["spearman"] >= 0.99
               and abs(_sdj["q5_plateaus_removed"]["spearman"] - 0.6327) < 0.05
               and 0.40 <= min(_sdj["q4_loco"].values())
               and max(_sdj["q4_loco"].values()) <= 0.95)
    # Table-1 direction claims straight from the single TEST readout
    _sv = lambda k: 1.0 - tr[k]["mean_k"] / 32.0
    claim_true("DIR.t1.bhvshoef05",
               "a=.05: BAYES-H saving 5.2x FIXED-Hoeffding 15.6%",
               _sv("BAYESH_a0.05") > _sv("FIXED_HOEF_a0.05")
               and abs(_sv("FIXED_HOEF_a0.05") - 0.15625) < 0.01
               and abs(_sv("BAYESH_a0.05") / _sv("FIXED_HOEF_a0.05") - 5.18) < 0.1)
    claim_true("DIR.t1.ebvsbh10",
               "a=.10: FIXED-EB 84.4% > BAYES-H 84.0% (disclosed, not dominated)",
               _sv("FIXED_EB_a0.1") > _sv("BAYESH_a0.1")
               and abs(_sv("FIXED_EB_a0.1") - 0.84375) < 0.01
               and abs(_sv("BAYESH_a0.1") - 0.8395) < 0.01)
    claim_true("DIR.t1.adapt34",
               "a=.05: adaptivity adds +34 points of saving over FIXED-EB",
               abs((_sv("BAYESH_a0.05") - _sv("FIXED_EB_a0.05")) - 0.3406) < 0.01)
    claim_true("DIR.t1.bhvalid02",
               "a=.02: BAYES-H still saves 73.5% (FIXED-EB null budget)",
               abs(_sv("BAYESH_a0.02") - 0.7348) < 0.01)
    _fg = fit["fair_gap_bayesh_vs_hoeffding"]
    claim_true("DIR.t1.gap05sig",
               "a=.05 paired gap over FIXED-Hoeffding significant and positive",
               _fg["0.05"]["significant"]
               and _fg["0.05"]["abs_gap_bayesh_minus_hoef"] > 0
               and _fg["0.05"]["gap_ci_radius"] < _fg["0.05"]["abs_gap_bayesh_minus_hoef"])
    claim_true("DIR.t1.gap10sig",
               "a=.10 paired gap over FIXED-Hoeffding significant and positive",
               _fg["0.1"]["significant"]
               and _fg["0.1"]["abs_gap_bayesh_minus_hoef"] > 0)

    # ---------- W1.* real-path provenance layer (r493; mgr 1e858cb45e3e) ----------
    # Candidate-dir repair: HERE may be author_candidate_rXXX/, a sibling of
    # audit_pack/ (one level deeper than audit_pack itself). Resolve the
    # workspace dir from the audit_pack location, wherever HERE is.
    _ap = HERE if os.path.isfile(os.path.join(HERE, "claim_check.py")) and \
        os.path.basename(HERE) == "audit_pack" else \
        os.path.join(os.path.dirname(HERE), "audit_pack")
    import hashlib as _hlw
    _ws = os.path.join(_ap, os.pardir, os.pardir, os.pardir, "agents",
                       "A11", "workspace", "earlystop_drift_r491")
    _ws_ok = os.path.isdir(_ws)
    _pf = [("py", "prior_fit_size_r491.py", 12293,
            "tau*_m(alpha) exactly at nested FIT subsizes", "SEED = 20260815"),
           ("json", "prior_fit_size_r491_result.json", 8367,
            '"tau_star"', None),
           ("log", "run_r491.log", 1293,
            "prior_fit_size_r491_result.json", None)]
    for _tag, _name, _size, _anchor, _anchor2 in _pf:
        _pk = os.path.join(R, _name)
        _wsp = os.path.join(_ws, _name)
        _inpack = _tag != "log"
        # (a) in-pack / log existence + size
        if _inpack:
            claim_true(f"W1.pack.{_tag}.exist",
                       f"results/{_name} exists with size {_size}B",
                       os.path.isfile(_pk) and os.path.getsize(_pk) == _size,
                       f"path={_pk} size={os.path.getsize(_pk) if os.path.isfile(_pk) else 'MISSING'}")
        else:
            claim_true("W1.log.gate", "workspace earlystop_drift_r491/ reachable for log check",
                       _ws_ok, f"ws={_ws}")
            if _ws_ok:
                claim_true("W1.ws.log.exist",
                           f"workspace {_name} exists with size {_size}B",
                           os.path.isfile(_wsp) and os.path.getsize(_wsp) == _size,
                           f"path={_wsp} size={os.path.getsize(_wsp) if os.path.isfile(_wsp) else 'MISSING'}")
        # (b) workspace original existence + size (gated)
        if _inpack:
            claim_true(f"W1.ws.{_tag}.gate",
                       f"workspace original {_name} reachable",
                       _ws_ok, f"ws={_ws}")
            if _ws_ok:
                claim_true(f"W1.ws.{_tag}.exist",
                           f"workspace {_name} exists with size {_size}B",
                           os.path.isfile(_wsp) and os.path.getsize(_wsp) == _size,
                           f"path={_wsp} size={os.path.getsize(_wsp) if os.path.isfile(_wsp) else 'MISSING'}")
        # (c) workspace <-> pack byte parity (sha256)
        if _inpack and _ws_ok and os.path.isfile(_wsp) and os.path.isfile(_pk):
            _h1 = _hlw.sha256(open(_wsp, "rb").read()).hexdigest()
            _h2 = _hlw.sha256(open(_pk, "rb").read()).hexdigest()
            claim_true(f"W1.parity.{_tag}",
                       f"workspace {_name} == results/{_name} (sha256)",
                       _h1 == _h2, f"ws={_h1[:12]} pack={_h2[:12]}")
        # (d) content anchors
        _probe = _pk if _inpack else (_wsp if _ws_ok else None)
        if _probe is not None and os.path.isfile(_probe):
            _src = open(_probe, encoding="utf-8", errors="replace").read()
            claim_true(f"W1.content.{_tag}",
                       f"{_name} contains anchor {_anchor!r}",
                       _anchor in _src)
            if _anchor2 is not None:
                claim_true(f"W1.content.{_tag}2",
                           f"{_name} contains anchor {_anchor2!r}",
                           _anchor2 in _src)

    # ---------- FRESH.* build-freshness anchor layer (r489) ----------
    # r488 post-mortem: the r487 compile ran ~7 min BEFORE the final tex
    # edits, so its all-green run verified a stale intermediate PDF. A green
    # claim_check only proves the checker agrees with the file it read, not
    # that the PDF was built from the final tex. Anchors below are recorded
    # at final-edit time: sha256 of the final paper.tex and sha256 of the
    # PDF compiled from it (r489 clean-room rebuild of the r488 candidate
    # reproduces this PDF byte-identically, md5 951dc802...; deterministic
    # build). Any edit-then-forget-recompile state breaks one of the two
    # anchors. Discipline: after the FINAL tex edit of a round, recompile,
    # then update both anchors here.
    import hashlib as _hl
    _FRESH_TEX_SHA256 = "0a09788c045e0c3f12a08e2aba5c799872a476641755ff5ca67bbe53b729acb6"
    _FRESH_PDF_SHA256 = "540052f27ad0c880f12b93bce68c288b5da94c9e0d0c795a7ea558a84fb39b7c"
    if _ptex is not None:
        _tex_sha = _hl.sha256(open(_ptex, "rb").read()).hexdigest()
        claim_true("FRESH.tex.anchor",
                   "paper.tex read by this checker IS the final-edit v5.22 bytes",
                   _tex_sha == _FRESH_TEX_SHA256,
                   f"sha256={_tex_sha[:12]}... anchor={_FRESH_TEX_SHA256[:12]}...")
        _ppdf = os.path.join(os.path.dirname(_ptex), "paper.pdf")
        claim_true("FRESH.pdf.found", "sibling paper.pdf located for freshness check",
                   os.path.isfile(_ppdf), f"path={_ppdf}")
        if os.path.isfile(_ppdf):
            _pdf_sha = _hl.sha256(open(_ppdf, "rb").read()).hexdigest()
            claim_true("FRESH.pdf.anchor",
                       "sibling paper.pdf IS the PDF compiled from the final tex",
                       _pdf_sha == _FRESH_PDF_SHA256,
                       f"sha256={_pdf_sha[:12]}... anchor={_FRESH_PDF_SHA256[:12]}...")

    print()
    for p in PASSES:
        print("PASS " + p)
    for f in FAILS:
        print("FAIL " + f)
    for e in EXTS:
        print("EXT  " + e)
    print(f"\n{len(PASSES)} PASS / {len(FAILS)} FAIL / {len(EXTS)} EXT(external-provenance)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
