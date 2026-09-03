"""pilot14: r779 unified-theory verification. Two arms.
Arm A (schedule/K invariance, convex): damage(always-on) identical across schedule shape
  (flat/mild/base, fixed E) and update count K in {1,10}; per-seed exact + 6-seed dist.
Arm B (SGD K-scan, falsifiable prediction of unified theory): SGD bs in {256,128,64,32}
  (K/epoch in {5,10,19,38}) => inj@210 frac DECREASES toward GD primacy value 0.087 as K drops.
  (pilot7 convex+SGD bs=128 K=9.4 gave 1.061; GD K=1 gave 0.087.)
Preregistered:
  P1 invariance: max |dmg| spread across schedule-shape cells < 0.01 AND K=1 vs K=10 < 0.01.
  P2 SGD K-scan: inj@210 frac monotone nonincreasing in K, and bs=256 (K=5) frac < bs=32 (K=38).
"""
import numpy as np, json, time
from pilot13_updatecount_vs_structure import sigmoid, convex_rep
from pilot7_ablate2x2 import convex_train, T, T_split
from pilot5_boundary import make_data
def acc(w,X,y): return float(np.mean((X@w>0)==(y>0.5)))

def train_gd_eta(XA,yA,XD,yD,ETA,lam,w0,always):
    w=w0.copy()
    for t in range(len(ETA)):
        blocks=[(XA,yA)]+([(XD,yD)] if always else [])
        X=np.concatenate([b[0] for b in blocks]); y=np.concatenate([b[1] for b in blocks])
        z=X@w; p=sigmoid(z); g=X.T@(p-y)/len(y)+lam*w
        w=w-ETA[t]*g
    return w

t0=time.time()
lam=1e-3
E_target=0.8*120+0.08*120
# Arm A: schedule shape invariance (GD K=1, always-on), 6 seeds
scheds={'base(0.8/0.08)':(0.8,0.08),'flat(0.44)':(0.44,0.44),'mild(0.6/0.28)':(0.6,0.28)}
armA={}
for name,(eh,el) in scheds.items():
    assert abs(eh*120+el*120-E_target)<1e-9
    ETA=np.array([eh if t<T_split else el for t in range(T)])
    dmgs=[]
    for seed in range(6):
        XA,yA,XD,yD,Xt,yt=make_data(seed)
        w0=np.random.default_rng(1000+seed).standard_normal(40)*0.05
        a_n=acc(train_gd_eta(XA,yA,XD,yD,ETA,lam,w0,False),Xt,yt)
        a_a=acc(train_gd_eta(XA,yA,XD,yD,ETA,lam,w0,True),Xt,yt)
        dmgs.append(a_n-a_a)
    armA[name]=float(np.median(dmgs))
# Arm A K: rep K=10 always-on
dmgs=[]
for seed in range(6):
    XA,yA,XD,yD,Xt,yt=make_data(seed)
    dmgs.append(acc(convex_rep(XA,yA,XD,yD,None,seed),Xt,yt)-acc(convex_rep(XA,yA,XD,yD,(0,T),seed),Xt,yt))
armA['rep(K=10)']=float(np.median(dmgs))
vals=list(armA.values())
P1=(max(vals)-min(vals))<0.01

# Arm B: SGD K-scan (convex), inj@210 frac
armB={}
full=None
for bs in [256,128,64,32]:
    fr=[]
    for seed in range(6):
        XA,yA,XD,yD,Xt,yt=make_data(seed)
        a_n=acc(convex_train(XA,yA,XD,yD,None,seed,sgd=True,bs=bs),Xt,yt)
        a_i=acc(convex_train(XA,yA,XD,yD,(210,T),seed,sgd=True,bs=bs),Xt,yt)
        a_a=acc(convex_train(XA,yA,XD,yD,(0,T),seed,sgd=True,bs=bs),Xt,yt)
        full_d=a_n-a_a
        fr.append((a_n-a_i)/full_d if full_d>1e-9 else 0.0)
    K=(600*2)//bs  # n total = 1200 (A+D when on); approx K per epoch
    armB[f'bs{bs}']={'frac_med':float(np.median(fr)),'K_approx':int(1200/bs)}
bs256=armB['bs256']['frac_med']; bs32=armB['bs32']['frac_med']
P2=(bs256<bs32)
out={'armA_invariance':armA,'P1_invariance':bool(P1),
     'armB_sgd_Kscan':armB,'P2_sgd_Kscan_decreasing':bool(P2),
     'GD_primacy_ref':0.087,'runtime_sec':round(time.time()-t0,1)}
json.dump(out,open('pilot14_invariance_out.json','w'),indent=1)
print(json.dumps(out,indent=1))
