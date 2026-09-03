import numpy as np, json, sys
sys.path.insert(0, '.')
from pilot13_updatecount_vs_structure import sigmoid

# convex cell, GD K=1, always-on D. vary (eta_hi, eta_lo) keeping E=105.6 fixed.
def make_data(seed):
    rng = np.random.default_rng(seed)
    d = 40; nA = 600; mu = np.zeros(d); mu[0] = 1.0; sep = 2.2
    XA = rng.standard_normal((nA, d)); yA = (rng.random(nA) < 0.5).astype(float)
    XA += (2*yA[:,None]-1)*(sep/2)*mu
    XD = XA + 0.05*rng.standard_normal((nA, d)); yD = 1.0-yA
    nt = 4000
    Xt = rng.standard_normal((nt,d)); yt = (rng.random(nt)<0.5).astype(float)
    Xt += (2*yt[:,None]-1)*(sep/2)*mu
    return XA,yA,XD,yD,Xt,yt

def acc(w,X,y): return float(np.mean((X@w>0)==(y>0.5)))

def train_gd(XA,yA,XD,yD,ETA,T,lam,w0,always):
    w = w0.copy()
    for t in range(T):
        blocks = [(XA,yA)] + ([(XD,yD)] if always else [])
        X = np.concatenate([b[0] for b in blocks]); y = np.concatenate([b[1] for b in blocks])
        z = X@w; p = sigmoid(z)
        g = X.T@(p-y)/len(y) + lam*w
        w = w - ETA[t]*g
    return w

T, T_split, lam = 240, 120, 1e-3
E_target = 0.8*120 + 0.08*120  # 105.6
configs = {
  'base(0.8/0.08)': (0.8, 0.08),
  'flat(E/T=0.44)': (0.44, 0.44),
  'mild(0.6/0.28)': (0.6, 0.28),
}
out = {}
for name,(eh,el) in configs.items():
    assert abs(eh*120+el*120 - E_target) < 1e-9, name
    ETA = np.array([eh if t<T_split else el for t in range(T)])
    dmgs = []
    for seed in range(6):
        XA,yA,XD,yD,Xt,yt = make_data(seed)
        rng = np.random.default_rng(1000+seed); w0 = rng.standard_normal(40)*0.05
        a_never = acc(train_gd(XA,yA,XD,yD,ETA,T,lam,w0,False),Xt,yt)
        a_always = acc(train_gd(XA,yA,XD,yD,ETA,T,lam,w0,True),Xt,yt)
        dmgs.append(a_never-a_always)
    out[name] = {'E': E_target, 'dmg_per_seed': dmgs, 'dmg_med': float(np.median(dmgs))}
    print(name, 'dmg_med=%.4f'%out[name]['dmg_med'], 'per-seed', [round(x,3) for x in dmgs])
json.dump(out, open('/tmp/diag_seam_out.json','w'), indent=1)
