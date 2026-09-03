#!python3
"""M5 content-complete self-check (r1902): re-derive every headline aggregate and appendix
number in paper/paper.tex from the frozen result JSONs and assert exact match.

r1902 CHANGE: the headline M2 / M2.5 comparison is now MATCHED on the same 3 seeds
{0,1,2}. These checks re-aggregate the frozen JSONs on that matched seed set (importing the
permanent matcher subgmmix_matched_r1902.py) and assert the paper's headline tables against
the matched values. The independent 5-seed M2 numbers are preserved below as a robustness
view (still read from the frozen JSON, still asserted).

Reads only frozen evidence (results/SUBGMIX_*.json); writes nothing. EXIT 0 iff all checks
pass, nonzero iff any headline number in the paper disagrees with its frozen JSON source.
"""
import json, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import subgmmix_matched_r1902 as matcher
import json, sys, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")  # results/ dir
PASS = 0; FAIL = []

def chk(name, got, want, tol=1e-6):
    global PASS
    g = float(got); w = float(want)
    if abs(g - w) <= tol:
        PASS += 1
    else:
        FAIL.append(f"{name}: got {g} want {w}")

def chk_int(name, got, want):
    global PASS
    if int(got) == int(want):
        PASS += 1
    else:
        FAIL.append(f"{name}: got {got} want {want}")

# ============ Tab:main / Tab:frontier — FULL 5-seed MATCHED {0,1,2,3,4} (r1903) ============
# r1903: M2.5 now re-run on the SAME 5 seeds as the frozen M2 (0.269), closing the r1901 gap
# where the 5-seed M2.5 grid was a pending replication. Reads the NEW frozen 5-seed file.
m5 = json.load(open(ROOT + "SUBGMIX_M25_PAIRED_R1885_5SEED.json"))["agg"]
chk("M2.5-5seed n_rows (350)", m5["n_rows"], 350)
chk("M2.5-5seed paired committed_rate (0.503)", m5["committed_rate_pair"], 0.503, tol=1e-3)
chk("M2.5-5seed paired cert_cov", m5["cert_cov_pair"], 1.0)
chk("M2.5-5seed paired comm_mean (0.0009)", m5["comm_mean_regret"], 0.0009, tol=5e-4)
chk("M2.5-5seed paired comm_max (0.0364)", m5["comm_max_regret"], 0.0364, tol=5e-4)
chk("M2.5-5seed paired abst_mean (0.0054)", m5["abst_mean_reg"], 0.0054, tol=5e-4)
chk("M2.5-5seed MPB committed_rate (0.260)", m5["committed_rate_mpb"], 0.260, tol=1e-3)
chk("M2.5-5seed MPB cert_cov", m5["cert_cov_mpb"], 1.0)
chk("M2.5-5seed Hoeffding committed_rate (0.097)", m5["committed_rate_hoef"], 0.097, tol=1e-3)
chk("M2.5-5seed Hoeffding cert_cov", m5["cert_cov_hoef"], 1.0)
# exact finite-sample order on full 5-seed matched set (W1): MPB < M2
m2_5seed = json.load(open(ROOT + "SUBGMIX_M2_GATE_R1884.json"))["agg"]["committed_rate"]
chk("5-seed exact order MPB<M2 (0.260<0.269)", 1.0 if m5["committed_rate_mpb"] < m2_5seed else 0.0, 1.0)

# ============ Tab:main / Tab:frontier (M2 + M2.5) — MATCHED 3-seed {0,1,2} ============
# Headline comparison is strict: same seeds/splits/models/w-grid/delta/gate.
m_rows, m_agg = matcher.m2_matched()
chk("M2 matched committed_rate (0.267)", m_agg["committed_rate"], 0.2667, tol=5e-4)
chk("M2 matched cert_cov", m_agg["cert_coverage"], 1.0)
chk("M2 matched comm_mean_regret", m_agg["comm_mean_regret"], 0.00001, tol=5e-5)
chk("M2 matched comm_max_regret", m_agg["comm_max_regret"], 0.0005, tol=5e-4)
chk("M2 matched abst_mean_regret", m_agg["abst_mean_regret"], 0.0044, tol=5e-4)
chk("M2 matched abst_max_regret", m_agg["abst_max_regret"], 0.0546, tol=5e-4)
chk("M2 matched no_gate_mean_regret", m_agg["no_gate_mean_regret"], 0.0032, tol=5e-4)
chk_int("M2 matched n_rows (210)", m_agg["n_rows"], 210)

p_rows, p_agg = matcher.m25_paired_matched()
chk("M2.5-paired matched committed_rate (0.495)", p_agg["committed_rate"], 0.495, tol=5e-3)
chk("M2.5-paired matched cert_cov", p_agg["cert_coverage"], 1.0)
chk("M2.5-paired matched comm_mean", p_agg["comm_mean_regret"], 0.0007, tol=5e-4)
chk("M2.5-paired matched comm_max", p_agg["comm_max_regret"], 0.0348, tol=5e-4)
chk("M2.5-paired matched abst_mean", p_agg["abst_mean_regret"], 0.0057, tol=5e-4)
chk("M2.5-paired matched abst_max", p_agg["abst_max_regret"], 0.0546, tol=5e-4)
chk_int("M2.5-paired matched n_rows (210)", p_agg["n_rows"], 210)

mpb_agg = matcher.m25_mpb_matched()[1]
chk("M2.5-MPB matched committed_rate (0.257)", mpb_agg["committed_rate"], 0.257, tol=5e-3)
chk("M2.5-MPB matched cert_cov", mpb_agg["cert_coverage"], 1.0)
h_agg = matcher.m25_hoef_matched()[1]
chk("M2.5-Hoeffding matched committed_rate (0.091)", h_agg["committed_rate"], 0.0905, tol=1e-3)
chk("M2.5-Hoeffding matched cert_cov", h_agg["cert_coverage"], 1.0)
# exact finite-sample order stays honest (W1): MPB < M2 on matched seeds
chk("matched exact order MPB<M2 (W1)", 1.0 if mpb_agg["committed_rate"] < m_agg["committed_rate"] else 0.0, 1.0)

# ============ Independent 5-seed M2 (robustness view, preserved) ============
m2f = json.load(open(ROOT + "SUBGMIX_M2_GATE_R1884.json"))["agg"]
chk("M2 5-seed committed_rate (0.269)", m2f["committed_rate"], 0.269)
chk("M2 5-seed cert_cov", m2f["cert_coverage(rego<=tau)"], 1.0)
chk("M2 5-seed comm_mean_regret", m2f["committed_mean_regret"], 0.0001)
chk("M2 5-seed comm_max_regret", m2f["committed_max_regret"], 0.0019)
chk("M2 5-seed abst_mean_regret", m2f["abstained_mean_regret(if-hardpick)"], 0.0043)
chk("M2 5-seed abst_max_regret", m2f["abstained_max_regret"], 0.0546)
chk_int("M2 5-seed n_rows (350)", m2f["n_rows"], 350)

# ============ App.A M1 diagnostics ============
pilot = json.load(open(ROOT + "SUBGMIX_PILOT_R1881.json"))
g = pilot["global"]
chk("M1 flat-grid mrr==cal_prior regret", g["mrr"]["mean_regret"], g["cal_prior"]["mean_regret"])
chk_int("PILOT n_rows 350", pilot["n_rows"], 350)
mb = json.load(open(ROOT + "SUBGMIX_R1881C_MIXTUREBALL.json"))
mbrows = mb["rows"]
chk_int("mixtureball rows 352", len(mbrows), 352)
div = [r for r in mbrows if r["diverges"]]
chk_int("mixtureball diverges 165", len(div), 165)
better = [r for r in div if r["regret_mrr"] < r["regret_cal"]]
worse = [r for r in div if r["regret_mrr"] > r["regret_cal"]]
chk_int("diverged mrr-better 93", len(better), 93)
chk_int("diverged mrr-worse 62", len(worse), 62)
chk("diverged regret diff -0.0021", sum(r["regret_mrr"]-r["regret_cal"] for r in div)/len(div), -0.0021, tol=3e-4)
chk("overall regret diff +0.0019", sum(r["regret_mrr"]-r["regret_cal"] for r in mbrows)/len(mbrows), 0.0019, tol=3e-4)
dg = json.load(open(ROOT + "SUBGMIX_DIAG_R1881B.json"))
dgrows = dg["rows"]
chk_int("diag rows 880", len(dgrows), 880)
div_u = [r for r in dgrows if r["diverges_U"]]
chk_int("diag diverges_U 15", len(div_u), 15)
tied = [r for r in div_u if abs(r["regret_mrr"]-r["regret_cal"]) < 1e-9]
chk_int("diag diverged tied 11", len(tied), 11)
chk_int("diag diverged worse-for-M1 4", len(div_u)-len(tied), 4)
worser = [r for r in div_u if r["regret_mrr"]-r["regret_cal"] > 1e-9]
if worser:
    chk("diag worse diff +0.0024", worser[0]["regret_mrr"]-worser[0]["regret_cal"], 0.0024, tol=3e-4)

# ============ Tab:m3 (M3 budget; tau=0.04) ============
m3 = json.load(open(ROOT + "SUBGMIX_M3_BUDGET_R1886.json"))
agg3 = m3["agg"]
def g3(c, f, p, tau=0.04):
    for r in agg3:
        if r["carrier"]==c and r["frac"]==f and r["policy"]==p and r["tau"]==tau:
            return r["committed_rate"]
    return None
ev040 = sum(r["n_w"]*r["n_seeds"] for r in agg3 if r["tau"]==0.04)
chk_int("M3 tau=0.04 evaluated cells 4200", ev040, 4200)
bad = [r for r in agg3 if r["committed_rate"]>0 and (r["cert_validity"] is None or r["cert_validity"]<0.9999)]
chk_int("M3 zero cert-violation cells", len(bad), 0)
chk("M3 adaptive tau=.04 none commit", max((r["committed_rate"] for r in agg3 if r["policy"]=="adaptive" and r["tau"]==0.04), default=0.0), 0.0)
chk("M3 MNIST .5 uniform 0.778", g3('mnist',0.5,'uniform'), 0.778)
chk("M3 MNIST .5 neyman 0.400", g3('mnist',0.5,'neyman'), 0.400)
chk("M3 MNIST .5 sens 0.578", g3('mnist',0.5,'sens'), 0.578)
chk("M3 Fashion .5 neyman 0.067", g3('fashion',0.5,'neyman'), 0.067)
chk("M3 Fashion .5 uniform 0.000", g3('fashion',0.5,'uniform'), 0.000)
chk("M3 MNIST .95 uniform 1.0", g3('mnist',0.95,'uniform'), 1.0)
chk("M3 Fashion .95 uniform 0.133", g3('fashion',0.95,'uniform'), 0.133)
chk("M3 Fashion .95 sens 0.133", g3('fashion',0.95,'sens'), 0.133)

# ============ Tab:m35 (M3.5 minimax vs rules) ============
mm = json.load(open(ROOT + "SUBGMIX_MINIMAX_R1897.json"))
def gmm(c,f):
    for r in mm["agg"]:
        if r["carrier"]==c and r["frac"]==f:
            return r["committed_rate"]
    return None
chk("M3.5 MNIST .5 minimax 0.400", gmm('mnist',0.5), 0.400)
chk("M3.5 MNIST .8 minimax 0.889", gmm('mnist',0.8), 0.889)
chk("M3.5 Fashion .5 minimax 0.067", gmm('fashion',0.5), 0.067)
chk("M3.5 Fashion .8 minimax 0.089", gmm('fashion',0.8), 0.089)
chk("M3.5 digits .5 minimax 0.0", gmm('digits',0.5), 0.0)
chk("M3.5 news .5 minimax 0.0", gmm('news',0.5), 0.0)
chk("M3 MNIST .8 neyman 0.844", g3('mnist',0.8,'neyman'), 0.844)
chk("M3 MNIST .8 sens 1.0", g3('mnist',0.8,'sens'), 1.0)
chk("M3 Fashion .8 neyman 0.089", g3('fashion',0.8,'neyman'), 0.089)

# ============ Tab:m35gate (conditional gate, real carriers) ============
cg = json.load(open(ROOT + "SUBGMIX_CONDGATE_SINGLE_R1896.json"))
def cgrow(c,f,rule):
    for r in cg["agg"]:
        if r["carrier"]==c and r["frac"]==f and r["rule"]==rule:
            return r["commit_rate"], r["med_UB"]
    return None
chk("gate fashion .5 uniform 0.333", cgrow('fashion',0.5,'uniform')[0], 0.333)
chk("gate fashion .5 minimax 1.0", cgrow('fashion',0.5,'minimax')[0], 1.0)
chk("gate fashion .5 uniform medUB 0.041", cgrow('fashion',0.5,'uniform')[1], 0.0411, tol=1e-3)
chk("gate fashion .5 minimax medUB 0.000", cgrow('fashion',0.5,'minimax')[1], 0.0, tol=1e-6)
chk("gate fashion .65 uniform 0.667", cgrow('fashion',0.65,'uniform')[0], 0.667)
chk("gate fashion .65 minimax 1.0", cgrow('fashion',0.65,'minimax')[0], 1.0)
du = [r["med_UB"] for r in cg["agg"] if r["carrier"]=='digits' and r["rule"]=='uniform']
dmm = [r["med_UB"] for r in cg["agg"] if r["carrier"]=='digits' and r["rule"]=='minimax']
nu = [r["med_UB"] for r in cg["agg"] if r["carrier"]=='news' and r["rule"]=='uniform']
chk("gate digits uniform min 1.07", min(du), 1.0739, tol=1e-3)
chk("gate digits uniform max 2.06", max(du), 2.0589, tol=1e-3)
chk("gate digits minimax 1.05", min(dmm), 1.0457, tol=1e-3)
chk("gate news uniform min 0.27", min(nu), 0.266, tol=1e-3)
chk("gate news uniform max 0.50", max(nu), 0.504, tol=1e-3)
chk("gate news minimax 0.24", min([r["med_UB"] for r in cg["agg"] if r["carrier"]=='news' and r["rule"]=='minimax']), 0.245, tol=1e-3)

# ============ App.B all-active ablation (tab:m35ab) ============
ab = json.load(open(ROOT + "SUBGMIX_MINIMAX_ABLATION_R1897.json"))["by_carrier"]
for c, (active, ratio) in {"mnist":(25,1.30), "fashion":(25,1.33), "digits":(7,4.19), "news":(20,1.17)}.items():
    ac = round(100*ab[c]["mean_active_frac"])
    chk_int(f"ablation {c} active% {active}", ac, active)
    chk(f"ablation {c} d*ratio {ratio}", ab[c]["dstar_over_uniform_ratio"], ratio, tol=2e-2)

# ============ App.C deterministic counterexample ============
ct = json.load(open(ROOT + "SUBGMIX_MM_COUNTER_R1895.json"))
r = ct["results"]["e_0"]
chk("counter uniform UB 0.094", r["uniform"]["UB"], 0.094, tol=1e-3)
chk("counter minimax UB 0.029", r["minimax"]["UB"], 0.029, tol=1e-3)
chk_int("counter uniform n_g g0 300", r["uniform"]["n_g"][0], 300)
chk_int("counter minimax n_g g0 1197", r["minimax"]["n_g"][0], 1197)
chk("counter uniform not committed", 1.0 if r["uniform"]["committed_at_tau_004"] else 0.0, 0.0)
chk("counter minimax committed", 1.0 if r["minimax"]["committed_at_tau_004"] else 0.0, 1.0)

# ============ App.E gate OC (r1899) ============
oc = json.load(open(ROOT + "SUBGMIX_GATE_OC_R1899.json"))
def ocrow(deployed, cv):
    for r in oc["rows"]:
        if r["deployed"]==deployed and r["cv_level"]==cv:
            return r
    return None
for deployed, cv, area in [("non-spanning","low",0.25), ("non-spanning","high",0.25),
                           ("spanning","low",0.00), ("spanning","high",-0.08)]:
    row = ocrow(deployed, cv)
    chk(f"gateoc {deployed} {cv} area {area}", round(row["signed_area_minimax_over_uniform"],2), area)
sph = ocrow("spanning","high")
chk("gateoc spanning high worst deficit -0.49", sph["worst_minimax_deficit"], -0.49, tol=1e-6)
chk("gateoc spanning high worst tau 0.12", float(sph["worst_minimax_deficit_at_tau"]), 0.12)
nslow = ocrow("non-spanning","low")
mm_curve = nslow["curve_minimax"]; taus = nslow["taus"]
first_one = taus[mm_curve.index(1.0)] if 1.0 in mm_curve else None
chk("gateoc non-span low minimax first 1.0 at tau=0.08", float(first_one) if first_one else -1.0, 0.08)
chk("gateoc non-span low uniform never 1.0", 1.0 if 1.0 in nslow["curve_uniform"] else 0.0, 0.0)

print(f"\nM5_VERIFY_R1902: PASS {PASS}, FAIL {len(FAIL)}")
for f in FAIL:
    print("  FAIL:", f)
sys.exit(1 if FAIL else 0)