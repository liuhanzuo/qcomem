#!/usr/bin/env python3
"""r790 final: verify every pilot/diag/verify-anchored numeric claim in paper.tex
against its *_out.json. One runnable check; exits 1 on any real mismatch.
Numbers with no artifact anchor (proof-skeleton asymptotics) are out of scope.
"""
import json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
# JSON artifacts live in the A2 workspace code dir; override via env when this
# script runs from the run-root paper/ copy (r792+).
JDIR = os.environ.get("CONSISTENCY_JDIR") or os.path.join(os.path.dirname(BASE), "lr_phase_datavalue_r1")
def load(f): return json.load(open(os.path.join(JDIR, f)))

ok = fail = 0
def chk(desc, claim, actual, tol=5e-4):
    global ok, fail
    if actual is None or (isinstance(actual, float) and actual != actual):  # None/NaN
        fail += 1; print(f"MISS  {desc}: actual={actual}"); return
    if abs(claim - actual) <= tol:
        ok += 1
    else:
        fail += 1; print(f"FAIL  {desc}: paper={claim} json={actual}")

# ---- fig1a / sec2: pilot2 + pilot3 ----
p2, p3 = load("pilot2_conflict_out.json"), load("pilot3_mechanism_out.json")
chk("pilot2 damage", 0.3395, p2["damage"])
chk("pilot2 inject_late_frac", 0.087, p2["inject_late_frac"])
chk("fig1a R2", 0.971, p3["r2_inject"])
chk("fig1a c_hat", 0.10266, p3["c_hat"], 1e-5)
for s, v, e in p3["inject"]:
    ref = {0:1.0,30:1.0,60:0.999,90:0.996,120:0.73,150:0.576,180:0.302,210:0.087,239:0.003}
    if s in ref: chk(f"fig1a inject@{s}", ref[s], v)
for s, v, r in p3["drop_resid"]:
    ref = {0:0.0,30:-0.001,60:-0.003,90:-0.012,120:-0.029,150:-0.031,180:-0.044,210:-0.05,239:-0.006}
    if s in ref: chk(f"fig1a drop@{s}", ref[s], v)

# ---- pilot1 (no object) ----
p1 = load("pilot_phase_datavalue_out.json")
chk("pilot1 damage ~ -0.008", -0.008, p1["damage"])

# ---- pilot10 conflict-strength sweep (sec2 + sec6.1) ----
p10 = load("pilot10_noise_strength_out.json")["per_rho"]
chk("p10 rho1.0 acc_rec", 1.031, p10["rho=1.0"]["acc_rec_drop120"], 1e-3)   # >=0.883 claim uses min
chk("p10 rho0.25 acc_rec (weakest)", 0.883, p10["rho=0.25"]["acc_rec_drop120"], 1e-3)
chk("p10 rho0.25 loss_rec", 0.732, p10["rho=0.25"]["loss_rec_drop120"], 1e-3)
chk("p10 weakest_erasable_rho", 0.25, load("pilot10_noise_strength_out.json")["weakest_erasable_rho"], 0)
acc_recs = [p10[f"rho={r}"]["acc_rec_drop120"] for r in ["1.0","0.75","0.5","0.25"]]
loss_recs = [p10[f"rho={r}"]["loss_rec_drop120"] for r in ["1.0","0.75","0.5","0.25"]]
chk("p10 min acc_rec>=0.883", True, min(acc_recs) >= 0.883 - 1e-3, 0)
chk("p10 min loss_rec>=0.732", True, min(loss_recs) >= 0.732 - 1e-3, 0)
chk("p10 rho0.1 no object", False, p10["rho=0.1"]["object_exists"], 0)

# ---- T1: pilot14 + verify_t1 ----
p14 = load("pilot14_invariance_out.json")
chk("p14 base damage", 0.34575, p14["armA_invariance"]["base(0.8/0.08)"], 1e-4)
chk("p14 flat damage", 0.34575, p14["armA_invariance"]["flat(0.44)"], 1e-4)
chk("p14 mild damage", 0.34575, p14["armA_invariance"]["mild(0.6/0.28)"], 1e-4)
chk("p14 rep(K=10) damage", 0.34612, p14["armA_invariance"]["rep(K=10)"], 1e-4)
chk("p14 P1", True, p14["P1_invariance"], 0)
chk("p14 P2 Kscan monotone", True, p14["P2_sgd_Kscan_decreasing"], 0)
chk("p14 GD primacy anchor", 0.087, p14["GD_primacy_ref"], 1e-3)
ks = p14["armB_sgd_Kscan"]
chk("p14 bs256 frac", 0.84, ks["bs256"]["frac_med"], 1e-2)
chk("p14 bs128 frac", 1.03, ks["bs128"]["frac_med"], 1e-2)
chk("p14 bs64 frac", 1.16, ks["bs64"]["frac_med"], 1e-2)
chk("p14 bs32 frac", 1.46, ks["bs32"]["frac_med"], 1e-2)

# ---- T2: pilot8 / pilot9 / diag5-via-pilot8 ----
p8 = load("pilot8_lambda_eff_out.json")
chk("p8 lam_eff_med", 0.0373, p8["lam_eff_med"], 1e-3)
chk("p8 lam_eff/ridge floor", 37.3, p8["lam_eff_over_lam"], 0.1)
chk("p8 E120 budget", 0.358, p8["lam_eff_x_E120"], 1e-3)
chk("p8 linear-explains-2.6% (0.358/13.816)", 0.026, p8["lam_eff_x_E120"]/13.816, 1e-3)
p9 = load("pilot9_param_resid_out.json")
chk("p9 param_resid@120", 0.0685, p9["pr120"], 1e-3)
chk("p9 loss_rec@120", 1.017, p9["lr120"], 1e-3)
chk("p9 acc_rec@120", 1.001, p9["ar120"], 1e-3)

# ---- T3: pilot15 / pilot16 / pilot12 / pilot13 / diag8 ----
p15 = load("pilot15_budget_flip_out.json")
chk("p15 A0 frac", 0.057, p15["A0_GD_base"]["frac_med"], 1e-3)
chk("p15 A1 frac", 0.922, p15["A1_GD_hightail"]["frac_med"], 1e-3)
chk("p15 A2 frac", 1.004, p15["A2_SGD_base"]["frac_med"], 1e-3)
chk("p15 A3 frac", 0.046, p15["A3_SGD_lowtail"]["frac_med"], 1e-3)
chk("p15 ratio A3/A0", 0.82, p15["verdicts"]["ratio_A3_over_A0"], 1e-2)
for k in ["P1_GD_flip_recency","P2_SGD_flip_primacy","P3_equal_budget_equal_absorption"]:
    chk(f"p15 {k}", True, p15["verdicts"][k], 0)
p16 = load("pilot16_mlp_budget_out.json")
chk("p16 MLP base frac (recency)", 0.94, p16["B0_base"]["frac_med"], 1e-2)
chk("p16 MLP budget-cut frac", 0.027, p16["B2_lo"]["frac_med"], 1e-3)
chk("p16 Q1", True, p16["verdicts"]["Q1_budget_down_primacy"], 0)
p12 = load("pilot12_noise_ablate_out.json")
chk("p12 fixedcyc inj210 (noise NOT cause)", 0.981, p12["modes"]["fixedcyc"]["inj210_frac"], 1e-3)
chk("p12 V1 falsified", False, p12["V1_noise_drives_flip"], 0)
p13 = load("pilot13_updatecount_vs_structure_out.json")
chk("p13 convex rep inj210 (K>1 alone->recency)", 0.907, p13["convex"]["rep"]["inj210_frac"], 1e-3)
d8 = load("diag8_stationary_cov_out.json")["rows"]
radii = [d8[b]["dist_runs_to_wstar_med"] for b in d8]
chk("diag8 fluct radius in [0.05,0.08]", True, all(0.04 <= r <= 0.09 for r in radii), 0)
chk("diag8 radius 30-50x < R=2.61", True, 2.61/max(radii) >= 30, 0)

# ---- lampath (no-param) ----
t3 = load("verify_t3_ou_out.json")
chk("lampath no-param", 0.2023, t3["lam_path_param_free"], 1e-3)

# ---- real-image arms: pilot17/18/19 (canonical full-6-seed JSONs) ----
# NOTE: pilot1718_verdict.json is partial (p17 n=4, p18 n=3, source=log-partial) and STALE;
# the full 6-seed JSONs are authoritative. sec6.2 uses the full-6-seed values.
import statistics as st
def arm_stats(f):
    d = load(f); aa = d["acc_all"]
    never, always = st.median(aa["never"]), st.median(aa["always"])
    drop, injl = st.median(aa["drop@120"]), st.median(aa["inject@late"])
    dmg = never - always
    return dmg, (drop-always)/dmg, (injl-always)/dmg
d17 = arm_stats("pilot17_realconflict_full_out.json")
d18 = arm_stats("pilot18_smallconflict_full_out.json")
d19 = arm_stats("pilot19_cifar10n_full_out.json")
chk("sec6.2 p17 damage", 0.0696, d17[0], 1e-3); chk("sec6.2 p17 rec", 1.014, d17[1], 1e-3)
chk("sec6.2 p17 frac_late", 0.894, d17[2], 1e-3)
chk("sec6.2 p18 damage", 0.0109, d18[0], 1e-3); chk("sec6.2 p18 rec", 1.069, d18[1], 1e-3)
chk("sec6.2 p18 frac_late", 0.479, d18[2], 1e-3)
chk("sec6.2 p19 damage", 0.0427, d19[0], 1e-3); chk("sec6.2 p19 rec", 0.240, d19[1], 1e-3)
chk("sec6.2 p19 frac_late", -0.345, d19[2], 1e-3)
# R1/R2/R3 verdicts unchanged on full data
chk("p17 R1", True, d17[0] >= 0.01, 0); chk("p17 R2", True, d17[1] >= 0.7, 0); chk("p17 R3 fails", True, d17[2] < 0.9, 0)
chk("p19 R1", True, d19[0] >= 0.01, 0); chk("p19 R2 FAILS", True, d19[1] < 0.7, 0)
# salvage (eqra_loss) + cos negative + P3
el = load("eqra_loss_full_out.json")["acc_median"]
chk("salvage eqra-loss", 0.6514, el["eqra-loss"], 1e-3)
chk("salvage drop@120", 0.6271, el["drop@120"], 1e-3)
chk("salvage +0.0243", 0.0243, el["eqra-loss"]-el["drop@120"], 1e-3)
chk("salvage never oracle", 0.6597, el["never"], 1e-3)
chk("salvage diff_vs_never -0.0083", -0.0083, el["eqra-loss"]-el["never"], 1e-3)
cs = load("eqra_salvage_full_out.json")["acc_median"]
chk("cos eqra-hard", 0.6156, cs["eqra-hard"], 1e-3)
chk("cos eqra-soft", 0.6218, cs["eqra-soft"], 1e-3)
chk("cos hard -0.013 vs drop", -0.013, cs["eqra-hard"]-cs["drop@120"], 1e-3)
p3f = load("eqra_loss_p3_full_out.json")["acc_median"]
chk("P3 always", 0.989, p3f["always"], 1e-3)
chk("P3 eqra-loss", 0.9627, p3f["eqra-loss"], 1e-3)
chk("P3 -0.026", -0.026, p3f["eqra-loss"]-p3f["always"], 1e-3)

# ---- diag20 Table 2 (incl. the r790 fixed cell) ----
d20 = load("diag20_traj_full_out.json")
chk("diag20 never acc", 0.6547, d20["acc_never"], 1e-3)
chk("diag20 always acc", 0.6091, d20["acc_always"], 1e-3)
chk("diag20 drop@120 acc", 0.6234, d20["traj"]["drop@120"]["test_acc"][-1], 1e-3)
chk("diag20 inject@late acc", 0.5996, d20["traj"]["inject@late"]["test_acc"][-1], 1e-3)
dloss = {30:[4.26,2.22,2.22,4.22],60:[7.74,0.50,0.49,7.74],120:[12.13,0.01,0.01,12.13],
         180:[13.31,0.00,0.05,13.31],239:[13.56,0.00,0.06,1.48]}
arms = ["never","always","drop@120","inject@late"]
for ep, vals in dloss.items():
    for a, vv in zip(arms, vals):
        chk(f"tab2 d_loss {a}@{ep}", vv, d20["traj"][a]["d_loss"][ep], 5e-3)
cos = {9:[0.08,0.08,0.15],69:[0.33,0.39,0.31],119:[-0.44,-0.37,0.04],
       189:[-0.90,-0.25,0.08],239:[-0.96,-0.04,-0.63]}   # 189 inject@late fixed to +0.08
for ep, vals in cos.items():
    for a, vv in zip(["always","drop@120","inject@late"], vals):
        chk(f"tab2 cos {a}@{ep}", vv, d20["cos"][a][str(ep)], 5e-3)
# fixed-drop grid recovery
dg = d20["drop_recovery"]
chk("diag20 grid max recovery (s=90)", 0.406, dg["90"], 1e-3)
chk("diag20 grid s=210", 0.171, dg["210"], 1e-3)
chk("diag20 grid anti-monotone (no s>=0.7)", True, all(x < 0.7 for x in dg.values()), 0)

# ---- r793: appendix-B enrichment anchors (proof numbers now in paper) ----
# diag9 commutator (T1 Step 3)
d9 = load("diag9_commutator_out.json")
chk("appB diag9 exact range 1.2e-4", 0.00012, d9["damage_range_exact"], 1e-5)
chk("appB diag9 frozenH range 1.2e-4", 0.00012, d9["damage_range_frozenH"], 1e-5)
chk("appB diag9 h-variation 0.0", 0.0, d9["h_variation_contribution"], 1e-9)
chk("appB diag9 C1 frozen<=exact", True, d9["C1_frozen_le_exact"], 0)
# Lemma C1 direct verification (10 points, min margin)
c1 = load("verify_c1_direct_out.json")
chk("appB C1 pass", True, c1["c1_direct_pass"], 0)
chk("appB C1 min margin 0.0392", 0.0392, min(r["margin"] for r in c1["rows"]), 1e-4)
chk("appB C1 n=10 points", 10, len(c1["rows"]), 0)
chk("appB C1 rhs~0.001", True, all(r["rhs"] < 0.002 for r in c1["rows"]), 0)
chk("appB C1 lam_e0 endpoints 0.036->0.51 (14.3x)", 14.3,
    max(r["lam_e0"] for r in c1["rows"])/min(r["lam_e0"] for r in c1["rows"]), 0.2)
# T3 OU param-free residuals + equal-budget matching
chk("appB t3 resid B2.4 A0 -0.085", -0.085, t3["resid_param_free"]["A0_GD_base"], 1e-3)
chk("appB t3 resid B2.4 A3 -0.078", -0.078, t3["resid_param_free"]["A3_SGD_lowtail"], 1e-3)
chk("appB t3 resid B24 within 0.007", True,
    abs(t3["resid_param_free"]["A1_GD_hightail"]) <= 0.007 and
    abs(t3["resid_param_free"]["A2_SGD_base"]) <= 0.007, 0)
chk("appB t3 equal-B match 0.0067", 0.0067,
    abs(t3["budgets"]["A0_GD_base"]["cov_meas"]-t3["budgets"]["A3_SGD_lowtail"]["cov_meas"]), 1e-4)
chk("appB t3 equal-B match 0.0002", 0.0002,
    abs(t3["budgets"]["A1_GD_hightail"]["cov_meas"]-t3["budgets"]["A2_SGD_base"]["cov_meas"]), 1e-4)
chk("appB t3 cov A0 0.470", 0.470, t3["budgets"]["A0_GD_base"]["cov_meas"], 1e-3)
chk("appB t3 cov A1 0.999", 0.999, t3["budgets"]["A1_GD_hightail"]["cov_meas"], 2e-3)
# pilot17 budget-mid arms (monotone across seam; deep-crossing under-prediction)
bm = load("pilot17_budget_mid_out.json")
chk("appB mid M1 cov 0.249", 0.249, bm["M1_B1.2"]["cov_med"], 1e-3)
chk("appB mid M2 cov 0.864", 0.864, bm["M2_B6.0"]["cov_med"], 1e-3)
chk("appB mid M2 pred 0.703", 0.703, bm["M2_B6.0"]["pred_cov_param_free"], 1e-3)
chk("appB mid M2 resid -0.161", -0.161, bm["M2_B6.0"]["resid"], 1e-3)
chk("appB mid monotone verdict", True, bm["verdicts"]["M_ord_monotone_across_seam"], 0)
chk("appB mid M_fit FAILS (disclosed)", False, bm["verdicts"]["M_fit_param_free"], 0)

print(f"\n=== r790+r793 consistency: {ok} pass, {fail} fail ===")
sys.exit(1 if fail else 0)
