"""
Phase-1 pilot11 (Gate B tension-1 closure): measure the LOCAL curvature lambda_eff ALONG
the erasure trajectory. This is the quantitative foundation of the Prop2 nonlinear
self-accelerating contraction lemma.

Context (pilot8/9, convex+GD cell): damage direction curvature at the clean endpoint
  lambda_eff(w_never) = 0.0373 (=37x ridge floor), tail budget E(120->T)=9.6.
  LINEAR prediction: resid_pred = exp(-0.0373*9.6) = 0.70.
  MEASURED param residual at drop@120: 0.0685  -> 10x more contraction than linear.
Candidate lemma: resid(s->T) = prod_t (1 - eta_t * lambda_eff(Delta_t)), where
  lambda_eff(Delta) = u^T H(w_never + Delta*u) u  along the damage direction u, and
  lambda_eff GROWS with ||Delta|| (logistic curvature is larger away from w_never in
  the +-mu direction), so the contraction self-accelerates.

Measurements (6 seeds, drop@s in {120,180,215}):
  Sim-arm (closure): START FROM w_s (train with D present up to s, STOP at s), replay the
    post-drop clean tail, each epoch record
      Delta_t = ||w_t - w_never||,  lam_t = u^T H(w_never + Delta_t u) u,
      factor_t = 1 - eta_t * lam_t   (clipped at 0 for safety in the product)
    then compare  prod factor_t   vs  actual ||w_T - w_never|| / ||w_s - w_never||.
  Quad-arm (ablation): same replay but lambda fixed at lambda_eff(w_never) (linear Prop2).
  Sweep-arm (shape): lambda_eff(Delta) on a grid Delta in [0, 1.2*||w_always-w_never||]
    to see how fast curvature grows with deviation.

Preregistered verdicts (median over seeds):
  V1 (mechanism confirmed): |prod_sim / resid_meas - 1| <= 0.5   (product law quantitative)
  V2 (self-acceleration):   lam(Delta_full) / lam(0) >= 1.5     (curvature grows with damage)
  V3 (linear falsified):    prod_quad / resid_meas >= 2         (linear law over-predicts resid)
"""
import numpy as np, json, time
from pilot5_boundary import make_data
from pilot7_ablate2x2 import convex_train, sigmoid, T, T_split

ETA_HI, ETA_LO, LAM = 0.8, 0.08, 1e-3


def eta_of_pw(t):
    return ETA_HI if t < T_split else ETA_LO


def hess_u(H_fn, w, u):
    return H_fn(w) @ u


def hessian(X, y, w, lam=LAM):
    z = X @ w
    p = sigmoid(z)
    r = p * (1.0 - p) / len(y)
    return (X.T * r) @ X + lam * np.eye(X.shape[1])


def train_until_s(XA, yA, XD, yD, s, seed):
    """D present for t<s, STOP at t=s (returns w_s, NOT a full-T run)."""
    rng = np.random.default_rng(20_000 + seed)
    w = rng.standard_normal(XA.shape[1]) * 0.05
    for t in range(s):
        eta = eta_of_pw(t)
        X = np.concatenate([XA, XD]); y = np.concatenate([yA, yD])
        z = X @ w
        g = X.T @ (sigmoid(z) - y) / len(y) + LAM * w
        w = w - eta * g
    return w


def main():
    t0 = time.time()
    seeds = list(range(6))
    drop_list = [120, 180, 215]
    per_s = {s: dict(prod_sim=[], prod_quad=[], resid=[], lam0=[], lamfull=[], ratio_lam=[],
                     neg_frac=[])
             for s in drop_list}
    shape_rows = []
    for sd in seeds:
        XA, yA, XD, yD, Xt, yt = make_data(sd)
        w_never = convex_train(XA, yA, XD, yD, None, sd, sgd=False)
        w_always = convex_train(XA, yA, XD, yD, (0, T), sd, sgd=False)
        d_full = np.linalg.norm(w_always - w_never)
        u = (w_always - w_never) / max(d_full, 1e-12)
        Hn = lambda w: hessian(XA, yA, w)
        lam0 = float(u @ Hn(w_never) @ u)
        lamfull = float(u @ Hn(w_never + d_full * u) @ u)
        # shape sweep (only for seed 0, save runtime; curvature path is smooth)
        if sd == 0:
            for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.2]:
                lam = float(u @ Hn(w_never + frac * d_full * u) @ u)
                shape_rows.append(dict(frac=frac, lam=round(lam, 5),
                                       lam_over_lam0=round(lam / max(lam0, 1e-12), 3)))
        for s in drop_list:
            w = train_until_s(XA, yA, XD, yD, s, sd)   # w_s: D present [0,s), STOPPED at s
            d_s = float(np.linalg.norm(w - w_never))
            prod_sim, prod_quad = 1.0, 1.0
            neg_traj = 0
            trace = []
            for t in range(s, T):
                eta = eta_of_pw(t)
                # full-batch clean-GD step (one epoch == one step)
                z = XA @ w
                g = XA.T @ (sigmoid(z) - yA) / len(yA) + LAM * w
                Delta = float(np.linalg.norm(w - w_never))
                dirv = (w - w_never) / max(Delta, 1e-12)
                lam_t = float(dirv @ Hn(w_never + Delta * dirv) @ dirv)
                f_sim = max(1.0 - eta * lam_t, 0.0)
                prod_sim *= f_sim
                if f_sim == 0.0:
                    neg_traj += 1   # entered the saturation region; linear factor would be <0
                prod_quad *= max(1.0 - eta * lam0, 0.0)
                w = w - eta * g
                if sd == 0 and s == 120 and t % 10 == 0:
                    trace.append(dict(t=t, Delta=round(Delta, 4), lam_t=round(lam_t, 4),
                                      eta_lam=round(eta * lam_t, 3)))
            per_s[s]["neg_frac"].append(neg_traj / (T - s))
            if sd == 0 and s == 120:
                per_s[s]["trace"] = trace
            resid = float(np.linalg.norm(w - w_never)) / max(d_s, 1e-12)
            per_s[s]["prod_sim"].append(prod_sim)
            per_s[s]["prod_quad"].append(prod_quad)
            per_s[s]["resid"].append(resid)
            per_s[s]["lam0"].append(lam0)
            per_s[s]["lamfull"].append(lamfull)
            per_s[s]["ratio_lam"].append(lamfull / max(lam0, 1e-12))
    med = lambda v: float(np.median(v))
    trace120 = per_s[120].get("trace", [])
    table = []
    for s in drop_list:
        r = per_s[s]
        table.append(dict(
            s=s,
            resid_meas=round(med(r["resid"]), 4),
            prod_sim=round(med(r["prod_sim"]), 4),
            prod_quad=round(med(r["prod_quad"]), 4),
            ratio_sim_over_meas=round(med(r["prod_sim"]) / max(med(r["resid"]), 1e-9), 3),
            ratio_quad_over_meas=round(med(r["prod_quad"]) / max(med(r["resid"]), 1e-9), 3),
            lam0=round(med(r["lam0"]), 5),
            lamfull=round(med(r["lamfull"]), 5),
            ratio_lam=round(med(r["ratio_lam"]), 3),
            neg_frac=round(med(r["neg_frac"]), 3),
        ))
    r120 = per_s[120]
    V1 = abs(med(r120["prod_sim"]) / max(med(r120["resid"]), 1e-9) - 1.0) <= 0.5
    V2 = med(r120["ratio_lam"]) >= 1.5
    V3 = med(r120["prod_quad"]) / max(med(r120["resid"]), 1e-9) >= 2.0
    out = dict(table=table, shape_sweep_seed0=shape_rows, trace_seed0_s120=trace120,
               V1_product_law_quantitative=bool(V1),
               V2_self_acceleration=bool(V2),
               V3_linear_falsified=bool(V3),
               runtime_sec=round(time.time() - t0, 1))
    print(json.dumps(out, indent=1))
    json.dump(out, open("pilot11_lambda_traj_out.json", "w"), indent=1)


if __name__ == "__main__":
    main()
