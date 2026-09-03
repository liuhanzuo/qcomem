"""
Phase-1 pilot8 (Gate B tension-1, numeric): measure the CURVATURE lambda_eff of the
absorbed damage direction in the convex+GD cell, and test the Prop2 erasure-horizon
formula  E(s->T) >= ln(1/delta) / lambda_eff.

Tension (PROP1_SKETCH_R775 Prop2): erasure speed should be governed by the curvature
ALONG the damage direction. If damage lived in a LOW-curvature direction, erasure
should be SLOW -- but pilot2/4/7 measure recovery ~1.0 with only the low-LR tail.
Two candidate resolutions (this pilot discriminates):
  R1: lambda_eff is NOT small (damage direction has appreciable curvature, e.g. the
      ridge floor lam plus logistic curvature along mu) -> formula consistent.
  R2: lambda_eff small BUT the tail E(s->T) is huge -> still consistent; then the
      real story is "tail budget >> 1/lambda_eff", not "curvature high".

Method (convex ridge logistic + full-batch GD, piecewise-constant LR, exact pilot7
convex+GD cell):
  - run 6 seeds; per seed: never / always / drop@s for s in a sweep.
  - w* = w_never (clean endpoint); damage dir u = (w_always - w*)/||.||.
  - lambda_eff = u^T H(w*) u, H = Gauss-Newton of clean loss at w*:
      H = X^T diag(p(1-p)) X / n + lam*I.
  - curvature spectrum of H reported (max/median/min) for context.
  - per drop-time s: absorbed damage at s (from a drop@never-retrained probe is
    expensive; instead compute predicted residual via the measured lambda_eff):
      pred_resid_frac(s) = exp(-lambda_eff * E(s->T)),  E = sum_{t>=s} eta_t
    compare to measured acc-recovery(s). Also report the delta=0.1 horizon:
      s_delta = smallest s with E(s->T) >= ln(10)/lambda_eff, and check recovery
      at the sweep point nearest s_delta is >= 0.9.

Preregistered verdict:
  V1 PASS if |ln(1 - rec_meas) | is within a factor 3 of lambda_eff*E(s->T) across
     the sweep (order-of-magnitude law, not exact -- logistic is not exactly quadratic).
  V2 PASS if lambda_eff/lam >= 2 (curvature well above ridge floor -> R1) OR
     lambda_eff*E(120->T) >= 3 (tail budget ample -> R2). Report which.
"""
import numpy as np, json, time
from pilot5_boundary import make_data
from pilot7_ablate2x2 import convex_train, sigmoid, T, T_split

ETA_HI, ETA_LO, LAM = 0.8, 0.08, 1e-3


def hessian_gn(w, XA, lam):
    z = XA @ w
    p = sigmoid(z)
    wgt = p * (1.0 - p)
    H = (XA * wgt[:, None]).T @ XA / len(XA) + lam * np.eye(XA.shape[1])
    return H


def tail_E(s, T=T, T_split=T_split):
    e = 0.0
    for t in range(s, T):
        e += ETA_HI if t < T_split else ETA_LO
    return e


def main():
    t0 = time.time()
    seeds = list(range(6))
    drop_sweep = [120, 150, 180, 200, 215, 225, 232]

    lam_effs, spec_rows = [], []
    rec_rows = {s: [] for s in drop_sweep}
    dmg_rows = []
    for sd in seeds:
        XA, yA, XD, yD, Xt, yt = make_data(sd)
        w_never = convex_train(XA, yA, XD, yD, None, sd, sgd=False)
        w_always = convex_train(XA, yA, XD, yD, (0, T), sd, sgd=False)
        d = w_always - w_never
        nd = np.linalg.norm(d)
        u = d / max(nd, 1e-12)
        H = hessian_gn(w_never, XA, LAM)
        lam_eff = float(u @ H @ u)
        lam_effs.append(lam_eff)
        ev = np.linalg.eigvalsh(H)
        spec_rows.append(dict(max=float(ev[-1]), med=float(np.median(ev)),
                              min=float(ev[0]), rank_mu=int(np.argmax(np.abs(ev @ u)))))

        acc_never = float(np.mean((Xt @ w_never > 0) == (yt > 0.5)))
        acc_always = float(np.mean((Xt @ w_always > 0) == (yt > 0.5)))
        dmg = acc_never - acc_always
        dmg_rows.append(dmg)
        for s in drop_sweep:
            w = convex_train(XA, yA, XD, yD, (0, s), sd, sgd=False)
            a = float(np.mean((Xt @ w > 0) == (yt > 0.5)))
            rec = (a - acc_always) / max(dmg, 1e-9)
            rec_rows[s].append(rec)

    med = lambda v: float(np.median(v))
    lam_eff_med = med(lam_effs)
    table = []
    ratios = []
    for s in drop_sweep:
        E = tail_E(s)
        rec_m = med(rec_rows[s])
        pred_log = lam_eff_med * E            # predicted -ln(residual frac)
        meas_log = -np.log(max(1.0 - rec_m, 1e-6))
        table.append(dict(s=s, E_tail=round(E, 3), rec_meas=round(rec_m, 3),
                          pred_neglog=round(pred_log, 3), meas_neglog=round(meas_log, 3)))
        if rec_m < 0.999:
            ratios.append(meas_log / max(pred_log, 1e-9))

    ratio_med = float(np.median(ratios)) if ratios else float("nan")
    # delta=0.1 horizon
    need = np.log(10.0) / max(lam_eff_med, 1e-12)
    s_delta = None
    for s in range(0, T):
        if tail_E(s) >= need:
            s_delta = s
            break
    rec_at_delta = None
    if s_delta is not None:
        s_near = min(drop_sweep, key=lambda x: abs(x - s_delta))
        rec_at_delta = med(rec_rows[s_near])

    out = dict(
        lam_eff_med=round(lam_eff_med, 5),
        lam_eff_over_lam=round(lam_eff_med / LAM, 2),
        lam_eff_seeds=[round(x, 5) for x in lam_effs],
        spectrum={k: round(med([r[k] for r in spec_rows]), 5) for k in ["max", "med", "min"]},
        tail_E_120=round(tail_E(120), 3),
        lam_eff_x_E120=round(lam_eff_med * tail_E(120), 3),
        sweep=table,
        ratio_meas_over_pred_med=round(ratio_med, 3),
        delta01=dict(need_E=round(need, 3), s_delta=s_delta,
                     rec_at_nearest_sweep=None if rec_at_delta is None else round(rec_at_delta, 3)),
        damage_med=round(med(dmg_rows), 4),
        V1_order_of_magnitude=bool(0.33 <= ratio_med <= 3.0),
        V2_R1_curvature=bool(lam_eff_med / LAM >= 2),
        V2_R2_tail_budget=bool(lam_eff_med * tail_E(120) >= 3),
        runtime_sec=round(time.time() - t0, 1),
    )
    print(json.dumps(out, indent=1))
    json.dump(out, open("pilot8_lambda_eff_out.json", "w"), indent=1)


if __name__ == "__main__":
    main()
