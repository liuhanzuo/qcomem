"""
Phase-0 pilot v4: is the 'drop late => NET GAIN (residual < 0)' robust across seeds?
pilot2 single-seed showed drop residual ~ -0.05 (net gain). Verify across 12 seeds.
Preregistered: if median residual < 0 AND >= 9/12 seeds negative => net-gain is real
(report effect size); else net-gain is seed noise, claim only 'residual ~ 0'.
"""
import numpy as np, json

def sigmoid(z): return 0.5 * (1.0 + np.tanh(0.5 * z))
def train(segs, T, eta_hi, eta_lo, T_split, lam, w0):
    w = w0.copy()
    for t in range(T):
        eta = eta_hi if t < T_split else eta_lo
        gsum = np.zeros_like(w); ntot = 0
        for (s, e, X, y) in segs:
            if s <= t < e:
                z = X @ w; p = sigmoid(z)
                gsum += (X.T @ (p - y) / len(y)) * len(y); ntot += len(y)
        w = w - eta * (gsum / max(ntot, 1) + lam * w)
    return w
def acc(w, X, y): return float(np.mean((X @ w > 0) == (y > 0.5)))

d, T, T_split = 40, 240, 120
eta_hi, eta_lo, lam = 0.8, 0.08, 1e-3
nA, nt, sep = 600, 4000, 2.2
mu = np.zeros(d); mu[0] = 1.0
res_drop210, dmg_list = [], []
for seed in range(12):
    rng = np.random.default_rng(100 + seed)
    XA = rng.standard_normal((nA, d)); yA = (rng.random(nA) < 0.5).astype(float)
    XA += (2 * yA[:, None] - 1) * (sep / 2) * mu
    XD = XA + 0.05 * rng.standard_normal((nA, d)); yD = 1.0 - yA
    Xt = rng.standard_normal((nt, d)); yt = (rng.random(nt) < 0.5).astype(float)
    Xt += (2 * yt[:, None] - 1) * (sep / 2) * mu
    w0 = rng.standard_normal(d) * 0.05
    a_never = acc(train([(0, T, XA, yA)], T, eta_hi, eta_lo, T_split, lam, w0), Xt, yt)
    a_always = acc(train([(0, T, XA, yA), (0, T, XD, yD)], T, eta_hi, eta_lo, T_split, lam, w0), Xt, yt)
    a_drop210 = acc(train([(0, T, XA, yA), (0, 210, XD, yD)], T, eta_hi, eta_lo, T_split, lam, w0), Xt, yt)
    dmg = a_never - a_always
    resid = (a_never - a_drop210) / dmg if dmg > 1e-9 else float('nan')
    res_drop210.append(resid); dmg_list.append(dmg)

res = np.array(res_drop210)
out = dict(
    dmg_mean=round(float(np.mean(dmg_list)), 4),
    dmg_min=round(float(np.min(dmg_list)), 4),
    resid_per_seed=[round(float(r), 3) for r in res],
    resid_median=round(float(np.median(res)), 4),
    resid_mean=round(float(np.mean(res)), 4),
    frac_negative=round(float(np.mean(res < 0)), 3),
    net_gain_real=bool(np.median(res) < 0 and np.mean(res < 0) >= 0.75),
)
print(json.dumps(out, indent=1))
json.dump(out, open("pilot4_seedrobust_out.json", "w"), indent=1)
