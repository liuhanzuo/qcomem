import numpy as np, sys
sys.path.insert(0, '.')
from pilot13_updatecount_vs_structure import sigmoid

def make_data(seed):
    rng=np.random.default_rng(seed); d=40; nA=600; mu=np.zeros(d); mu[0]=1.0; sep=2.2
    XA=rng.standard_normal((nA,d)); yA=(rng.random(nA)<0.5).astype(float)
    XA+=(2*yA[:,None]-1)*(sep/2)*mu
    XD=XA+0.05*rng.standard_normal((nA,d)); yD=1.0-yA
    return XA,yA,XD,yD
def grad(X,y,w,lam):
    z=X@w; p=sigmoid(z); return X.T@(p-y)/len(y)+lam*w
XA,yA,XD,yD=make_data(0)
lam=1e-3; T=240; T_split=120
ETA=np.array([0.8 if t<T_split else 0.08 for t in range(T)])

# attractor for window [0,s): train on A+D for s steps
def attractor(s):
    w=np.random.default_rng(1000).standard_normal(40)*0.05
    Xc=np.concatenate([XA,XD]); yc=np.concatenate([yA,yD])
    for t in range(s): w=w-ETA[t]*grad(Xc,yc,w,lam)
    return w
# clean w*
w=np.random.default_rng(1000).standard_normal(40)*0.05
for t in range(T): w=w-ETA[t]*grad(XA,yA,w,lam)
w_star=w.copy()

# drop arm: on [0,s) then clean [s,T). track distance to w*_s^(D) during on-phase, then to w* during clean
for s in [30, 120]:
    w=np.random.default_rng(1000).standard_normal(40)*0.05
    Xc=np.concatenate([XA,XD]); yc=np.concatenate([yA,yD])
    # on phase
    for t in range(s): w=w-ETA[t]*grad(Xc,yc,w,lam)
    w_after_on=w.copy()
    w_att=attractor(s)
    print(f"s={s}: ||w(s)-w*_s^(D)||={np.linalg.norm(w_after_on-w_att):.4f}  ||w(s)-w*||={np.linalg.norm(w_after_on-w_star):.4f}")
    # clean phase: track contraction to w*
    d0=np.linalg.norm(w_after_on-w_star)
    for t in range(s,T): w=w-ETA[t]*grad(XA,yA,w,lam)
    dT=np.linalg.norm(w-w_star)
    print(f"   clean tail: ||w(T)-w*||={dT:.4f}  contraction={dT/d0:.4f}  E_tail={ETA[s:].sum():.1f}")
