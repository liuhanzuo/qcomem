"""
r778 Prop2 closed-form verification: logistic curvature along the data direction.

Checks Lemma C1: lam_u(w) = u^T H_A(w) u vs the e_0 closed form
  lam_e0(w) = lam_ridge + (1/n) sum_i q(w.x_i) * x_{i,0}^2 ,  q = sigmoid(1-sigmoid).

Reproduces the table in PROP2_FORMAL_R778.md §1 (seed 0, make_data(0), pilot11 Hessian).
"""
import numpy as np, json
from pilot5_boundary import make_data
from pilot7_ablate2x2 import convex_train, sigmoid, T

LAM = 1e-3


def hessian(X, y, w, lam=LAM):
    z = X @ w
    p = sigmoid(z)
    r = p * (1.0 - p) / len(y)
    return (X.T * r) @ X + lam * np.eye(X.shape[1])


def main():
    XA, yA, XD, yD, Xt, yt = make_data(0)
    w_never = convex_train(XA, yA, XD, yD, None, 0, sgd=False)
    w_always = convex_train(XA, yA, XD, yD, (0, T), 0, sgd=False)
    d_full = np.linalg.norm(w_always - w_never)
    u = (w_always - w_never) / d_full

    rows = []
    for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        w = w_never + frac * d_full * u
        lam_meas = float(u @ hessian(XA, yA, w) @ u)
        # e_0 closed form: collapse to the e_0 coordinate (w_never rest is small)
        z = w[0] * XA[:, 0]
        p = sigmoid(z)
        lam_e0 = float(np.mean(p * (1 - p) * XA[:, 0] ** 2)) + LAM
        rows.append(dict(frac=frac, lam_meas=round(lam_meas, 5),
                         lam_e0_closed=round(lam_e0, 5),
                         ratio=round(lam_meas / max(lam_e0, 1e-9), 3)))

    # Lemma C1 sanity: curvature lower bound strictly above ridge floor
    lam_at_wstar = rows[0]["lam_meas"]
    above_floor = lam_at_wstar > 10 * LAM
    # saturation: curvature grows then saturates (pilot11 shape)
    out = dict(table=rows,
               u_proj_e0=round(float(u[0]), 4),
               d_full=round(float(d_full), 4),
               wstar_e0=round(float(w_never[0]), 4),
               C1_above_ridge_floor=bool(above_floor),
               lam_at_wstar_over_floor=round(lam_at_wstar / LAM, 1))
    print(json.dumps(out, indent=1))
    json.dump(out, open("verify_prop2_closedform_out.json", "w"), indent=1)


if __name__ == "__main__":
    main()
