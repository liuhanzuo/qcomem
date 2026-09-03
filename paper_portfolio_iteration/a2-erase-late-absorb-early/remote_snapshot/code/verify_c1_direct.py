"""verify_c1_direct (r782, A4 Gate-B audit M3): DIRECT check of Lemma C1's lower bound.

C1 claims (statement & proof now unified at the tighter constant, r782):
    lam_e0(w) := e0^T H_A(w) e0  >=  lam_r + q(|zbar| + 2*sqrt(d-1)) * m^2
where H_A = Hessian of the CLEAN-block-A ridge logistic loss, q(z)=sigmoid(z)(1-sigmoid(z)),
|zbar| = (1/n) sum_i |w.x_i|  (mean abs logit over the n samples of block A, assumption A5),
m^2 = (sep/2)^2 (cluster-mean contribution to (1/n) sum x_{i,0}^2).

r778's table only compared lam_meas(u) vs lam_closed(e0) on DIFFERENT directions -- that
cannot validate a lower bound. Here we evaluate BOTH sides along e0, at w = w_never +
frac * (w_always - w_never) for frac in {0, .25, .5, .75, 1.0} (the damage-direction
sweep of pilot11), and record margin = LHS - RHS per frac. PASS iff min margin >= 0.
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


rows = []
for seed in [0, 1]:
    XA, yA, XD, yD, Xt, yt = make_data(seed)
    n, d = XA.shape
    e0 = np.zeros(d); e0[0] = 1.0
    w_never = convex_train(XA, yA, XD, yD, None, seed, False)     # D never present
    w_always = convex_train(XA, yA, XD, yD, (0, T), seed, False)   # D always present
    dvec = w_always - w_never
    m2 = 1.1 ** 2  # (sep/2)^2, sep=2.2
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w = w_never + frac * dvec
        # LHS: e0^T H e0 via Hessian-vector product
        lhs = float(e0 @ (hessian(XA, yA, w) @ e0))
        # RHS: lam_r + q(|zbar| + 2*sqrt(d-1)) * m^2
        z = XA @ w
        zbar = float(np.mean(np.abs(z)))
        thr = zbar + 2.0 * np.sqrt(d - 1)
        q = float(sigmoid(thr) * (1.0 - sigmoid(thr)))
        rhs = LAM + q * m2
        rows.append(dict(seed=seed, frac=frac, lam_e0=lhs, rhs=rhs,
                         margin=lhs - rhs, zbar=zbar))

margins = [r["margin"] for r in rows]
out = dict(rows=rows, min_margin=float(np.min(margins)),
           max_margin=float(np.max(margins)),
           c1_direct_pass=bool(np.min(margins) >= 0.0),
           note=("direct e0-direction check of Lemma C1 (r782). margin = e0^T H e0 - RHS. "
                 "PASS iff min margin >= 0 over frac sweep x seeds {0,1}."))
json.dump(out, open("verify_c1_direct_out.json", "w"), indent=1)
print(json.dumps(out, indent=1))
