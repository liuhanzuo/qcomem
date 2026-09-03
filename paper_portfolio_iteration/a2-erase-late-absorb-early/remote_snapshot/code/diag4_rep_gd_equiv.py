import numpy as np, sys
sys.path.insert(0, '.')
from pilot13_updatecount_vs_structure import sigmoid, convex_rep
from pilot7_ablate2x2 import convex_train, T, T_split
from pilot5_boundary import make_data

# 1) exact lambda(delta) along displacement from w*, fit law
# 2) equilibrium distance of always-on w_T from w* for GD vs rep
XA,yA,XD,yD,Xt,yt = make_data(0)
lam=1e-3

def grad(X,y,w):
    z=X@w; p=sigmoid(z); return X.T@(p-y)/len(y)+lam*w

# w* (clean GD)
ETA=np.array([0.8 if t<T_split else 0.08 for t in range(T)])
w_clean=w_star=None
w=np.random.default_rng(1000).standard_normal(40)*0.05
for t in range(T):
    w=w-ETA[t]*grad(XA,yA,w)
w_star=w.copy()
print("||w*||=%.4f"%np.linalg.norm(w_star))

# combined equilibrium (always on): GD
w=np.random.default_rng(1000).standard_normal(40)*0.05
Xc=np.concatenate([XA,XD]); yc=np.concatenate([yA,yD])
for t in range(T):
    w=w-ETA[t]*grad(Xc,yc,w)
d_gd=np.linalg.norm(w-w_star)
print("GD always-on ||w_T-w*||=%.4f"%d_gd)

# rep always-on
w_rep=convex_rep(XA,yA,XD,yD,(0,T),0)
d_rep=np.linalg.norm(w_rep-w_star)
print("rep always-on ||w_T-w*||=%.4f  (ratio rep/gd=%.3f)"%(d_rep,d_rep/d_gd))

# curvature along displacement direction u = (w_rep - w*) normalized
u=(w_rep-w_star)/d_rep
def dir_curv(w,u):
    z=XA@w; q=sigmoid(z)*(1-sigmoid(z))
    H=(XA*q[:,None]).T@XA/len(yA)+lam*np.eye(40)
    return float(u@H@u)
for frac in [0,0.25,0.5,0.75,1.0]:
    w=w_star+frac*d_rep*u
    print(f"frac={frac}: lambda_u={dir_curv(w,u):.5f}")
