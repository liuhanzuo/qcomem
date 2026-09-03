"""TABLE_FIG_VALUE_AUDIT_R841: per-value anchoring of every hardcoded number in
tables tab:budget / tab:diag20 and Figure 1 panels (a/b/c) against primary JSONs.

Prior dimensions covered: grid sizes (r836), figure MATH self-consistency (r819),
symbol/notation (r839), render/present integrity (r830). NONE anchored each concrete
numeric cell in the three tables / three figure panels to its primary JSON value.
This is that audit.

Protocol:
  L0 fault-injection on an in-memory copy proves the detectors fire.
  L1 tab:budget  (4 arms x frac + cov)     <- pilot15_budget_flip_out.json
  L2 fig(a) inject/drop curves (9+9 pts)   <- pilot3_mechanism_out.json
  L3 fig(b) 4 bars                          <- pilot14_invariance_out.json
  L4 fig(c) 4 pts + lambath                <- pilot15 + pilot11 trapezoid
  L5 tab:diag20 loss block (5 ep x 4 arms) <- diag20_traj_full_out.json
  L6 tab:diag20 cos block  (5 ep x 3 arms) <- diag20_traj_full_out.json
  L7 prose key values (damage/recovery)    <- pilot17/18/19 + diag20
exit 0 = all pass.
"""
import json, numpy as np, re, sys, copy

JDIR = '/newcpfs/user/qixuan1/01_p5/run/iclr27_theory_k3_20260806_r1/agents/A2/workspace/lr_phase_datavalue_r1/'
TEX = '/newcpfs/user/qixuan1/01_p5/run/iclr27_theory_k3_20260806_r1/agents/A2/workspace/paper/paper.tex'

p11 = json.load(open(JDIR + 'pilot11_lambda_traj_out.json'))
p13 = json.load(open(JDIR + 'pilot13_updatecount_vs_structure_out.json'))
p14 = json.load(open(JDIR + 'pilot14_invariance_out.json'))
p15 = json.load(open(JDIR + 'pilot15_budget_flip_out.json'))
p3 = json.load(open(JDIR + 'pilot3_mechanism_out.json'))
d20 = json.load(open(JDIR + 'diag20_traj_full_out.json'))
p17 = json.load(open(JDIR + 'pilot17_realconflict_full_out.json'))
p18 = json.load(open(JDIR + 'pilot18_smallconflict_full_out.json'))
p19 = json.load(open(JDIR + 'pilot19_cifar10n_full_out.json'))

TOL = 5e-3  # tex rounds to 3 decimals; allow rounding slack

fails = []
def ok(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond:
        fails.append(name)

def close(a, b, tol=TOL):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False

# ---------- expected tex values (hardcoded cells), kept literal so fault-injection can flip them
def build_tex():
    return {
        # L1 tab:budget  frac, cov
        'tab_A0_frac': 0.057, 'tab_A0_cov': 0.470,
        'tab_A1_frac': 0.922, 'tab_A1_cov': 0.999,
        'tab_A2_frac': 1.004, 'tab_A2_cov': 0.999,
        'tab_A3_frac': 0.046, 'tab_A3_cov': 0.463,
        # L3 fig(b) bars
        'figb_base': 0.34575, 'figb_flat': 0.34575, 'figb_mild': 0.34575, 'figb_rep': 0.34612,
        # L4 fig(c) pts
        'figc_GD_low': 0.0567, 'figc_GD_high': 0.9220,
        'figc_SGD_low': 0.0463, 'figc_SGD_high': 1.0044,
        'lampath_tex': 0.2023,
        # L5 diag20 loss rows (never/always/drop/inject) at ep [30,60,120,180,239]
        'd20loss': {
            30:  [4.26, 2.22, 2.22, 4.22],
            60:  [7.74, 0.50, 0.49, 7.74],
            120: [12.13, 0.01, 0.01, 12.13],
            180: [13.31, 0.00, 0.05, 13.31],
            239: [13.56, 0.00, 0.06, 1.48],
        },
        # L6 diag20 cos rows (always/drop/inject) at ep [9,69,119,189,239]
        'd20cos': {
            9:   [0.08, 0.08, 0.15],
            69:  [0.33, 0.39, 0.31],
            119: [-0.44, -0.37, 0.04],
            189: [-0.90, -0.25, 0.08],
            239: [-0.96, -0.04, -0.63],
        },
        # L7 prose real-image damage/recovery
        'p17_damage': 0.0696, 'p17_recovery': 1.014,
        'p18_damage': 0.0109, 'p18_recovery': 1.069,
        'p19_damage': 0.0427, 'p19_recovery': 0.240,
    }

def audit(tex, verbose=True):
    global fails
    fails = []
    # L1 tab:budget from pilot15
    arm = {'A0': p15['A0_GD_base'], 'A1': p15['A1_GD_hightail'],
           'A2': p15['A2_SGD_base'], 'A3': p15['A3_SGD_lowtail']}
    ok('L1 tab_A0_frac', close(tex['tab_A0_frac'], arm['A0']['frac_med']), f"{arm['A0']['frac_med']:.4f}")
    ok('L1 tab_A0_cov',  close(tex['tab_A0_cov'],  arm['A0']['cov_med']),  f"{arm['A0']['cov_med']:.4f}")
    ok('L1 tab_A1_frac', close(tex['tab_A1_frac'], arm['A1']['frac_med']), f"{arm['A1']['frac_med']:.4f}")
    ok('L1 tab_A1_cov',  close(tex['tab_A1_cov'],  arm['A1']['cov_med']),  f"{arm['A1']['cov_med']:.4f}")
    ok('L1 tab_A2_frac', close(tex['tab_A2_frac'], arm['A2']['frac_med']), f"{arm['A2']['frac_med']:.4f}")
    ok('L1 tab_A2_cov',  close(tex['tab_A2_cov'],  arm['A2']['cov_med']),  f"{arm['A2']['cov_med']:.4f}")
    ok('L1 tab_A3_frac', close(tex['tab_A3_frac'], arm['A3']['frac_med']), f"{arm['A3']['frac_med']:.4f}")
    ok('L1 tab_A3_cov',  close(tex['tab_A3_cov'],  arm['A3']['cov_med']),  f"{arm['A3']['cov_med']:.4f}")
    # ratio A3/A0
    ok('L1 ratio_A3/A0=0.82', close(p15['verdicts']['ratio_A3_over_A0'], 0.817, 1e-2),
       f"{p15['verdicts']['ratio_A3_over_A0']:.4f}")

    # L2 fig(a) curves from pilot3
    inj = {int(r[0]): r[1] for r in p3['inject']}
    drp = {int(r[0]): r[1] for r in p3['drop_resid']}
    tex_inj = {0:1.0,30:1.0,60:0.999,90:0.996,120:0.73,150:0.576,180:0.302,210:0.087,239:0.003}
    tex_drp = {0:0.0,30:-0.001,60:-0.003,90:-0.012,120:-0.029,150:-0.031,180:-0.044,210:-0.05,239:-0.006}
    for s, v in tex_inj.items():
        ok(f'L2 figa_inject@{s}', close(v, inj[s]), f"json={inj[s]}")
    for s, v in tex_drp.items():
        ok(f'L2 figa_drop@{s}', close(v, drp[s]), f"json={drp[s]}")
    ok('L2 figa_c_hat=0.10266', close(p3['c_hat'], 0.10266, 1e-4), f"{p3['c_hat']}")
    ok('L2 figa_R2=0.971', close(p3['r2_inject'], 0.971, 1e-2), f"{p3['r2_inject']:.4f}")

    # L3 fig(b) bars from pilot14
    a = p14['armA_invariance']
    ok('L3 figb_base', close(tex['figb_base'], a['base(0.8/0.08)']), f"{a['base(0.8/0.08)']:.5f}")
    ok('L3 figb_flat', close(tex['figb_flat'], a['flat(0.44)']), f"{a['flat(0.44)']:.5f}")
    ok('L3 figb_mild', close(tex['figb_mild'], a['mild(0.6/0.28)']), f"{a['mild(0.6/0.28)']:.5f}")
    ok('L3 figb_rep',  close(tex['figb_rep'],  a['rep(K=10)']),  f"{a['rep(K=10)']:.5f}")

    # L4 fig(c) points + lambath
    ok('L4 figc_GD_low',  close(tex['figc_GD_low'],  p15['A0_GD_base']['frac_med']),  f"{p15['A0_GD_base']['frac_med']:.4f}")
    ok('L4 figc_GD_high', close(tex['figc_GD_high'], p15['A1_GD_hightail']['frac_med']), f"{p15['A1_GD_hightail']['frac_med']:.4f}")
    ok('L4 figc_SGD_high',close(tex['figc_SGD_high'],p15['A2_SGD_base']['frac_med']),  f"{p15['A2_SGD_base']['frac_med']:.4f}")
    ok('L4 figc_SGD_low', close(tex['figc_SGD_low'], p15['A3_SGD_lowtail']['frac_med']), f"{p15['A3_SGD_lowtail']['frac_med']:.4f}")
    # lampath from pilot11 trapezoid
    rows = p11['shape_sweep_seed0']
    fr = np.array([r['frac'] for r in rows]); la = np.array([r['lam'] for r in rows])
    m = (fr >= 0.0) & (fr <= 1.0)
    lam_path = float(np.trapz(la[m], fr[m]))
    ok('L4 lampath=0.2023', close(tex['lampath_tex'], lam_path, 1e-3), f"trapezoid={lam_path:.4f}")

    # L5 diag20 loss block
    arms20 = ['never', 'always', 'drop@120', 'inject@late']
    for ep, texrow in tex['d20loss'].items():
        for i, armn in enumerate(arms20):
            arr = d20['traj'][armn]['d_loss']
            jv = arr[ep]
            ok(f'L5 d20loss[{ep}][{armn}]', close(texrow[i], jv, 2e-2), f"json={jv:.3f}")

    # L6 diag20 cos block (always/drop/inject)
    cosarms = ['always', 'drop@120', 'inject@late']
    for ep, texrow in tex['d20cos'].items():
        for i, armn in enumerate(cosarms):
            jv = d20['cos'][armn].get(str(ep))
            ok(f'L6 d20cos[{ep}][{armn}]', close(texrow[i], jv, 2e-2), f"json={jv:.3f}")

    # L7 prose real-image values
    for name, dd, dam, rec in [('p17', p17, tex['p17_damage'], tex['p17_recovery']),
                               ('p18', p18, tex['p18_damage'], tex['p18_recovery']),
                               ('p19', p19, tex['p19_damage'], tex['p19_recovery'])]:
        ok(f'L7 {name}_damage',   close(dam, dd['damage'], 1e-3),   f"json={dd['damage']}")
        ok(f'L7 {name}_recovery', close(rec, dd['recovery_drop_at_split'], 1e-2), f"json={dd['recovery_drop_at_split']}")
    # frac_inject_late prose (0.894 / 0.479 / -0.345)
    ok('L7 p17_frac_late=0.894', close(p17['frac_inject_late'], 0.894, 1e-3), f"{p17['frac_inject_late']}")
    ok('L7 p18_frac_late=0.479', close(p18['frac_inject_late'], 0.479, 1e-3), f"{p18['frac_inject_late']}")
    ok('L7 p19_frac_late=-0.345', close(p19['frac_inject_late'], -0.345, 1e-3), f"{p19['frac_inject_late']}")
    return fails

# ---- run
print("== TABLE_FIG_VALUE_AUDIT_R841 ==")
tex = build_tex()
audit(tex)
main_fails = list(fails)
print()
print("RESULT:", "ALL PASS" if not main_fails else f"{len(main_fails)} FAIL: {main_fails}")

# ---------- L0 fault injection (prove detectors fire)
print("\n== L0 fault-injection self-check ==")
bad = build_tex(); bad['tab_A0_cov'] = 0.999  # flip
f0 = audit(bad, verbose=False)
print("inject tab_A0_cov 0.470->0.999:", "DETECTED" if 'L1 tab_A0_cov' in f0 else "MISSED")
bad = build_tex(); bad['d20loss'][239] = [13.56, 9.99, 0.06, 1.48]
f0 = audit(bad, verbose=False)
print("inject d20loss[239] always 0.00->9.99:", "DETECTED" if 'L5 d20loss[239][always]' in f0 else "MISSED")
bad = build_tex(); bad['d20cos'][239] = [-0.96, -0.04, +0.99]
f0 = audit(bad, verbose=False)
print("inject d20cos[239] inject -0.63->+0.99:", "DETECTED" if 'L6 d20cos[239][inject@late]' in f0 else "MISSED")
bad = build_tex(); bad['lampath_tex'] = 0.9999
f0 = audit(bad, verbose=False)
print("inject lampath 0.2023->0.9999:", "DETECTED" if 'L4 lampath=0.2023' in f0 else "MISSED")

sys.exit(0 if not main_fails else 1)
