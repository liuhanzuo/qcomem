"""
Phase-0 pilot v3 (mechanism lock-in): damage = F(exposure in high-LR phase).

From pilot2: damage(frac) is monotone in inject time's overlap with high-LR.
Hypothesis (scalar law): with D injected at time s (window [s,T)),
  damage_frac(s) = 1 - exp(-c * E(s)),  E(s) = sum_{t>=s} eta_t  (cumulative LR after s)
Equivalently log(1-damage) linear in E(s). Also test drop-side:
  recovery(s) for window [0,s) should equal 1 - damage_frac_effective, i.e.
  residual damage = 1-exp(-c * E_early(s)) with E_early(s)=sum_{t<s} eta_t.

Preregistered:
  M1: R^2 of linear fit log(1-damage) ~ E(s) >= 0.95 on inject curve.
  M2: same constant c predicts drop-curve residual damage within 15% rel err.
  M3: zero-exposure limit: inject at s=T-1 (only last step) => damage < 5%.
"""
import numpy as np, json

def sigmoid(z): return 0.5 * (1.0 + np.tanh(0.5 * z))
def loss_grad(w, X, y, lam):
    z = X @ w; p = sigmoid(z)
    return None, X.T @ (p - y) / len(y) + lam * w
def train(segs, T, eta_hi, eta_lo, T_split, lam, w0):
    w = w0.copy()
    for t in range(T):
        eta = eta_hi if t < T_split else eta_lo
        gsum = np.zeros_like(w); ntot = 0
        for (s, e, X, y) in segs:
            if s <= t < e:
                _, g = loss_grad(w, X, y, 0.0)
                gsum += g * len(y); ntot += len(y)
        w = w - eta * (gsum / max(ntot, 1) + lam * w)
    return w
def acc(w, X, y): return float(np.mean((X @ w > 0) == (y > 0.5)))

rng = np.random.default_rng(0)
d, T, T_split = 40, 240, 120
eta_hi, eta_lo, lam = 0.8, 0.08, 1e-3
nA = 600; mu = np.zeros(d); mu[0] = 1.0; sep = 2.2
XA = rng.standard_normal((nA, d)); yA = (rng.random(nA) < 0.5).astype(float)
XA += (2 * yA[:, None] - 1) * (sep / 2) * mu
XD = XA + 0.05 * rng.standard_normal((nA, d)); yD = 1.0 - yA
nt = 4000
Xt = rng.standard_normal((nt, d)); yt = (rng.random(nt) < 0.5).astype(float)
Xt += (2 * yt[:, None] - 1) * (sep / 2) * mu
w0 = rng.standard_normal(d) * 0.05

def eta_of(t): return eta_hi if t < T_split else eta_lo
ETA = np.array([eta_of(t) for t in range(T)])
def E_after(s): return float(ETA[s:].sum())
def E_before(s): return float(ETA[:s].sum())

def sched(win):
    return [(0, T, XA, yA), (win[0], win[1], XD, yD)]

acc_never = acc(train([(0, T, XA, yA)], T, eta_hi, eta_lo, T_split, lam, w0), Xt, yt)
acc_always = acc(train(sched((0, T)), T, eta_hi, eta_lo, T_split, lam, w0), Xt, yt)
damage_full = acc_never - acc_always

# inject curve
drop_times = [0, 30, 60, 90, 120, 150, 180, 210, 239]
inj_s, inj_dmg, inj_E = [], [], []
for s in drop_times:
    a = acc(train(sched((s, T)), T, eta_hi, eta_lo, T_split, lam, w0), Xt, yt)
    frac = (acc_never - a) / damage_full
    inj_s.append(s); inj_dmg.append(frac); inj_E.append(E_after(s))

# fit log(1-damage) = -c * E  (through origin)
inj_dmg_c = np.clip(inj_dmg, 0, 0.9999)
y_fit = -np.log(1 - inj_dmg_c)
E_arr = np.array(inj_E)
c_hat = float((E_arr @ y_fit) / (E_arr @ E_arr))
pred = 1 - np.exp(-c_hat * E_arr)
ss_res = float(np.sum((np.array(inj_dmg) - pred) ** 2))
ss_tot = float(np.sum((np.array(inj_dmg) - np.mean(inj_dmg)) ** 2))
r2 = 1 - ss_res / max(ss_tot, 1e-12)

# drop curve: predict residual damage from E_before(s)
drop_s, drop_resid_meas, drop_resid_pred = [], [], []
for s in drop_times:
    a = acc(train(sched((0, s)), T, eta_hi, eta_lo, T_split, lam, w0), Xt, yt)
    resid_meas = (acc_never - a) / damage_full   # remaining damage
    resid_pred = 1 - np.exp(-c_hat * E_before(s))
    drop_s.append(s); drop_resid_meas.append(float(resid_meas)); drop_resid_pred.append(float(resid_pred))
rel_errs = [abs(m - p) / max(abs(m), 1e-6) for m, p in zip(drop_resid_meas, drop_resid_pred) if abs(m) > 0.02]
mean_rel = float(np.mean(rel_errs)) if rel_errs else 0.0

out = dict(
    damage_full=round(damage_full, 4),
    inject=[(s, round(dmg, 3), round(E, 1)) for s, dmg, E in zip(inj_s, inj_dmg, inj_E)],
    c_hat=round(c_hat, 5), r2_inject=round(r2, 4),
    drop_resid=[(s, round(m, 3), round(p, 3)) for s, m, p in zip(drop_s, drop_resid_meas, drop_resid_pred)],
    mean_rel_err_drop=round(mean_rel, 4),
    M1=bool(r2 >= 0.95), M2=bool(mean_rel <= 0.15),
    M3=bool(inj_dmg[-1] < 0.05),
)
print(json.dumps(out, indent=1))
json.dump(out, open("pilot3_mechanism_out.json", "w"), indent=1)
