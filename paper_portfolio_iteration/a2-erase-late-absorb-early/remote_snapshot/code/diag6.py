import numpy as np, sys
sys.path.insert(0, '.')
from pilot13_updatecount_vs_structure import sigmoid
from pilot7_ablate2x2 import convex_train, T, T_split
from pilot5_boundary import make_data
def acc(w,X,y): return float(np.mean((X@w>0)==(y>0.5)))
XA,yA,XD,yD,Xt,yt=make_data(0)
lam=1e-3
def grad(X,y,w):
    z=X@w; p=sigmoid(z); return X.T@(p-y)/len(y)+lam*w
ETA=np.array([0.8 if t<T_split else 0.08 for t in range(T)])
# clean w*
w=np.random.default_rng(1000).standard_normal(40)*0.05
for t in range(T): w=w-ETA[t]*grad(XA,yA,w)
w_star=w.copy()
# SGD always-on, inj@210, never — distances to w*
w_al=convex_train(XA,yA,XD,yD,(0,T),0,sgd=True)
w_in=convex_train(XA,yA,XD,yD,(210,T),0,sgd=True)
w_nv=convex_train(XA,yA,XD,yD,None,0,sgd=True)
print("SGD: ||w_never-w*||=%.4f  ||w_always-w*||=%.4f  ||w_inj210-w*||=%.4f"%(
    np.linalg.norm(w_nv-w_star),np.linalg.norm(w_al-w_star),np.linalg.norm(w_in-w_star)))
print("SGD acc: never=%.4f always=%.4f inj210=%.4f"%(acc(w_nv,Xt,yt),acc(w_al,Xt,yt),acc(w_in,Xt,yt)))
# GD equivalents
w_al_g=convex_train(XA,yA,XD,yD,(0,T),0,sgd=False)
w_in_g=convex_train(XA,yA,XD,yD,(210,T),0,sgd=False)
w_nv_g=convex_train(XA,yA,XD,yD,None,0,sgd=False)
print("GD:  ||w_never-w*||=%.4f  ||w_always-w*||=%.4f  ||w_inj210-w*||=%.4f"%(
    np.linalg.norm(w_nv_g-w_star),np.linalg.norm(w_al_g-w_star),np.linalg.norm(w_in_g-w_star)))
print("GD acc: never=%.4f always=%.4f inj210=%.4f"%(acc(w_nv_g,Xt,yt),acc(w_al_g,Xt,yt),acc(w_in_g,Xt,yt)))
# noise ball: run SGD never with 3 seeds, spread around w*
ds=[]
for seed in range(3):
    w=convex_train(XA,yA,XD,yD,None,seed,sgd=True)
    ds.append(np.linalg.norm(w-w_star))
print("SGD never ||w-w*|| across seeds:", [round(x,4) for x in ds])
