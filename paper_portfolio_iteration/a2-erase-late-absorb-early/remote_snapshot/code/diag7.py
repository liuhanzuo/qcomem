import numpy as np, sys
sys.path.insert(0,'.')
from pilot13_updatecount_vs_structure import sigmoid
from pilot7_ablate2x2 import convex_train, T, T_split
from pilot5_boundary import make_data
XA,yA,XD,yD,Xt,yt=make_data(0)
lam=1e-3
def grad(X,y,w):
    z=X@w; p=sigmoid(z); return X.T@(p-y)/len(y)+lam*w
ETA=np.array([0.8 if t<T_split else 0.08 for t in range(T)])
# clean w* and clean attractor noise floor under SGD bs
def clean_attractor_sgd(bs,seed):
    rng=np.random.default_rng(30000+seed)
    w=np.random.default_rng(1000).standard_normal(40)*0.05
    X,y=XA,yA
    for t in range(T):
        eta=ETA[t]
        perm=rng.permutation(len(y))
        for i in range(0,len(y),bs):
            idx=perm[i:i+bs]
            z=X[idx]@w; p=sigmoid(z); g=X[idx].T@(p-y[idx])/len(idx)+lam*w
            w=w-eta*g
    return w
w_star=np.random.default_rng(1000).standard_normal(40)*0.05
for t in range(T): w_star=w_star-ETA[t]*grad(XA,yA,w_star)
# inj210: on-phase last 30 steps noise floor (combined data)
def inj210_endpoint(bs,seed):
    rng=np.random.default_rng(30000+seed)
    w=np.random.default_rng(1000).standard_normal(40)*0.05
    for t in range(T):
        eta=ETA[t]
        on=(210<=t<T)
        blocks=[(XA,yA)]+([(XD,yD)] if on else [])
        X=np.concatenate([b[0] for b in blocks]); y=np.concatenate([b[1] for b in blocks])
        perm=rng.permutation(len(y))
        for i in range(0,len(y),bs):
            idx=perm[i:i+bs]
            z=X[idx]@w; p=sigmoid(z); g=X[idx].T@(p-y[idx])/len(idx)+lam*w
            w=w-eta*g
    return w
for bs in [256,128,64,32]:
    nA_,nD_=len(yA),len(yD)
    Kon=(nA_+nD_)//bs; Koff=nA_//bs
    # SGD effective per-step LR = eta; noise floor ~ eta*sqrt(K)*sigma. Use eta_lo=0.08
    eta_lo=0.08
    inj=inj210_endpoint(bs,0)
    # distance of inj endpoint from clean never endpoint
    nv=clean_attractor_sgd(bs,0)
    print(f"bs={bs} K_on={Kon} K_off={Koff}: ||inj210-w_never||={np.linalg.norm(inj-nv):.4f}  pred_noise~eta*sqrt(K_on)={eta_lo*np.sqrt(Kon):.4f}")
