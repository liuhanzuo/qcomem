"""
Phase-1 pilot9 (Gate B tension-1 followup): is the free-erasure a TRUE parameter-space
return to the clean attractor, or merely an acc-threshold shadow?

pilot8 showed: damage direction has lambda_eff = 0.0373 (37x ridge floor, R1), yet the
linear-quadratic contraction (I-eta H)^k predicts only -log(resid)=0.36 over the tail
E(120->T)=9.6, while measured acc recovery is ~1.0 (-log resid = 13.8). Linear Prop2
under-predicts by ~40x. Two candidate mechanisms:
  (a) strong-attractor: clean GD nonlinearly pulls w back into the basin of w*, so the
      PARAMETER residual ||w_drop - w_never|| -> 0 and loss also recovers.
  (b) threshold-shadow: a sizable parameter residual remains but lies along directions
      that do not cross the sign boundary, so acc recovers while loss/params do not.

This pilot measures, per drop@s (convex+GD cell, 6 seeds):
  param_resid(s)  = ||w_drop@s - w_never|| / ||w_always - w_never||   (1 = full damage, 0 = clean)
  loss_rec(s)     = (loss_always - loss_drop)/(loss_always - loss_never)  (test loss side)
  acc_rec(s)      = (acc_drop - acc_always)/(acc_never - acc_always)
  margin_resid(s) = mean signed clean-train margin of drop vs never (function-level)

Verdict:
  M_PARAM (a) if param_resid(120) <= 0.3 AND loss_rec(120) >= 0.7   -> prove parameter-level Prop2
  M_SHADOW (b) if param_resid(120) >= 0.6 AND acc_rec(120) >= 0.9    -> prove function-level Prop2 only
  else MIXED -> report curve, refine.
Also report the param_resid decay across s to see if even the LAST few low-LR steps erase
(params) or whether param erasure needs the whole tail.
"""
import numpy as np, json, time
from pilot5_boundary import make_data
from pilot7_ablate2x2 import convex_train, sigmoid, T, T_split

ETA_HI, ETA_LO, LAM = 0.8, 0.08, 1e-3


def test_loss(w, Xt, yt):
    z = Xt @ w
    return float(np.mean(np.logaddexp(0, z) - yt * z))


def main():
    t0 = time.time()
    seeds = list(range(6))
    drop_sweep = [120, 150, 180, 200, 215, 225, 232]
    rows = {s: dict(pr=[], lr=[], ar=[], mr=[]) for s in drop_sweep}
    for sd in seeds:
        XA, yA, XD, yD, Xt, yt = make_data(sd)
        w_never = convex_train(XA, yA, XD, yD, None, sd, sgd=False)
        w_always = convex_train(XA, yA, XD, yD, (0, T), sd, sgd=False)
        nd_full = np.linalg.norm(w_always - w_never)
        acc_never = float(np.mean((Xt @ w_never > 0) == (yt > 0.5)))
        acc_always = float(np.mean((Xt @ w_always > 0) == (yt > 0.5)))
        loss_never, loss_always = test_loss(w_never, Xt, yt), test_loss(w_always, Xt, yt)
        dmg = acc_never - acc_always
        marg_never = float(np.mean((2 * yA - 1) * (XA @ w_never)))
        for s in drop_sweep:
            w = convex_train(XA, yA, XD, yD, (0, s), sd, sgd=False)
            pr = np.linalg.norm(w - w_never) / max(nd_full, 1e-12)
            a = float(np.mean((Xt @ w > 0) == (yt > 0.5)))
            l = test_loss(w, Xt, yt)
            m = float(np.mean((2 * yA - 1) * (XA @ w)))
            rows[s]["pr"].append(pr)
            rows[s]["lr"].append((loss_always - l) / max(loss_always - loss_never, 1e-9))
            rows[s]["ar"].append((a - acc_always) / max(dmg, 1e-9))
            rows[s]["mr"].append((m - marg_never))
    med = lambda v: float(np.median(v))
    table = [dict(s=s,
                  param_resid=round(med(rows[s]["pr"]), 4),
                  loss_rec=round(med(rows[s]["lr"]), 3),
                  acc_rec=round(med(rows[s]["ar"]), 3),
                  margin_resid=round(med(rows[s]["mr"]), 3)) for s in drop_sweep]
    pr120 = med(rows[120]["pr"]); lr120 = med(rows[120]["lr"]); ar120 = med(rows[120]["ar"])
    verdict = "M_PARAM" if (pr120 <= 0.3 and lr120 >= 0.7) else \
              ("M_SHADOW" if (pr120 >= 0.6 and ar120 >= 0.9) else "MIXED")
    out = dict(sweep=table, pr120=round(pr120, 4), lr120=round(lr120, 3),
               ar120=round(ar120, 3), verdict=verdict,
               runtime_sec=round(time.time() - t0, 1))
    print(json.dumps(out, indent=1))
    json.dump(out, open("pilot9_param_resid_out.json", "w"), indent=1)


if __name__ == "__main__":
    main()
