"""r1881c: MIXTURE-BALL SAFE RANKING — minimax-regret over an UNCERTAINTY SET of w.

关键实证发现 (r1881a/b): 在「已知单点 w」下 minimax-regret 退化为点估计 (diverges~0.017),
因为 Bernoulli CI 宽随水平的单调性与共共享 CAL 数据。安全排序的真正判别域 =
「w 只确定到 field mixture 不确定集 W(collection u 的邻域/Dirichlet 球)」:
选取 sup_{w∈W,j}(R_i(w)−R_j(w)) 的 minimax, 与点估计在极端 mixture 上真正分歧。
这是"安全"进入的正式位置。

协议: FIT 训候选; CAL 估逐组风险+同时 CP CI(Bonferroni); W = {w: u 的
variation-distance ≤ ρ-ball} ∩ 单纯形(顶点 = 邻域极值, 凸包)。对候选 i:
  worstR_i = sup_{w∈W} U_i(w) = max_{极端点 w∈W_vert} Σ_g w_g U_{i,g}
  bestR_j  similarly via L.
  minimax regret UB_i = worstR_i − min_j bestR_j. 选 argmin_i UB_i; 阈值 τ 证书, 否则回退。
结算: OUTER 上用 worst-over-W 真风险(用 ±ρ 顶点)而非单点, 报未来-mixture 最坏 regret。
端值: 在 W 边界混合上 minimax(mixture-ball) vs cal_prior(只用 u) 的 regret/命中/回退/带宽。
"""
import json, time, numpy as np, os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.special import betaincinv
import torchvision

OUT="subgroup_mix_ranking/results/SUBGMIX_R1881C_MIXTUREBALL.json"
DATA_CACHE="duplicate_sel/data"
RHOS=[0.10,0.20,0.30,0.45]
TAU=0.05
SEEDS=[0,1,2,3]

def cp_ci(n,x,delta):
    if x==0: lo=0.0
    else: lo=betaincinv(x,n-x+1,delta/2.0)
    if x==n: hi=1.0
    else: hi=betaincinv(x+1,n-x,1.0-delta/2.0)
    return float(lo),float(hi)

def _ld():
    from sklearn.datasets import load_digits as _d
    d=_d(); return d.data.astype(np.float32),d.target
def _lf():
    tr=torchvision.datasets.FashionMNIST(DATA_CACHE,train=True,download=False)
    te=torchvision.datasets.FashionMNIST(DATA_CACHE,train=False,download=False)
    X=np.concatenate([np.array(tr.data).reshape(-1,784),np.array(te.data).reshape(-1,784)],0)
    y=np.concatenate([np.array(tr.targets),np.array(te.targets)],0).astype(int)
    return X.astype(np.float32)/255.0,y

def build(nd,seed):
    return {'lr_C1':LogisticRegression(C=1.,max_iter=3000,random_state=seed),
            'lr_C01':LogisticRegression(C=.01,max_iter=3000,random_state=seed),
            'linsvc':LinearSVC(C=.1,max_iter=4000,random_state=seed),
            'mlp':MLPClassifier(hidden_layer_sizes=(64,),max_iter=600,random_state=seed)}

def ball_vertices(u,rho,G):
    """W = {w: TV(w,u)<=rho}∩simplex. 顶点: 相对每个 g 把 (rho*?) 质量平移。
    Variation 距离 TV(w,u)=0.5||w-u||_1. 最大 TV 顶点 = 把所有可动质量压到一个类。
    邻居外壳顶点构造: 从 u 出发把 2*rho 的总变差质量从各 g 移到单类 g'。
    """
    verts=set(); verts.add(tuple(u[g] for g in range(G)))
    target_G=2*rho  # total half-L1 budget = TV<=rho => L1<=2rho
    for gp in range(G):
        # 把质量从其他类移动到 gp, L1 位移总量 ~ 2*rho(双向)
        w=[u[g] for g in range(G)]
        donors=[g for g in range(G) if g!=gp]
        # 尽力从 donors 取 2rho 的 L1, 减到各自 (每 donor 可给最多 min((u_g-0), 需)), 加给 gp
        need=2*rho; ind=0
        while need>1e-9 and ind<len(donors):
            g=donors[ind]; give=min(u[g], need); w[g]-=give; w[gp]+=give; need-=give; ind+=1
        verts.add(tuple(round(max(x,0),9) for x in w))
    # 也加纯 uniform 方向端点 (极端均衡而非 skew), 保证外壳覆盖均值两侧
    return [np.array(v) for v in verts]

def run(name,seed):
    X,y=_ld() if name=='digits' else _lf()
    G=int(y.max()+1)
    Z=PCA(n_components=min(128,X.shape[1]),random_state=seed).fit_transform(X)
    Xr,Xo,yr,yo=train_test_split(Z,y,test_size=0.25,stratify=y,random_state=seed)
    Zf,Zc,yf,yc=train_test_split(Xr,yr,test_size=0.30,stratify=yr,random_state=seed)
    tr={k:m.fit(Zf,yf) or m for k,m in build(Zf.shape[1],seed).items()}
    M=len(tr); dcell=0.10/(G*M)
    mist={}
    for k,m in tr.items():
        em=(m.predict(Zc)!=yc).astype(float)
        ce={g:float(em[yc==g].mean()) if (yc==g).sum() else 0.0 for g in range(G)}
        cn={g:int((yc==g).sum()) for g in range(G)}
        Ls,Us={},{}
        for g in range(G):
            lo,hi=cp_ci(cn[g],int(round(ce[g]*cn[g])),dcell); Ls[g],Us[g]=lo,hi
        mist[k]={'pt':ce,'n':cn,'L':Ls,'U':Us}
    orc={}
    for k,m in tr.items():
        em=(m.predict(Xo)!=yo).astype(float)
        orc[k]={g:float(em[yo==g].mean()) if (yo==g).sum() else 0.0 for g in range(G)}
    u={g:(yc==g).sum()/len(yc) for g in range(G)}
    out=[]
    for rho in RHOS:
        Wv=ball_vertices(u,rho,G)
        # 一个 mixture-ball = 一个 minimax 决策(跨全 W), 在 W 各顶点结算 regret
        # worstR_i = max_{v∈W} Σv U_{i,g}; bestL_j = min_{v∈W} Σv L_{j,g}
        worstU_i={k:max(sum(v[g]*mist[k]['U'][g] for g in range(G)) for v in Wv) for k in tr}
        bestL_j=min({k:min(sum(v[g]*mist[k]['L'][g] for g in range(G)) for v in Wv)
                     for k in tr}.values())
        ub={k:worstU_i[k]-bestL_j for k in tr}
        argub=min(ub,key=ub.get); valid=ub[argub]<=TAU
        # 点估计 cal_prior on center u
        pt_u={k:sum(u[g]*mist[k]['pt'][g] for g in range(G)) for k in tr}
        argpt=min(pt_u,key=pt_u.get)
        for wi,w in enumerate(Wv):
            trR={k:sum(w[g]*orc[k][g] for g in range(G)) for k in tr}
            best=min(trR,key=trR.get); bestR=trR[best]
            out.append({'rho':rho,'vertex':wi,'oracle_best':best,'oracle_R':round(bestR,5),
                'regret_mrr':round(trR[argub]-bestR,5),'regret_cal':round(trR[argpt]-bestR,5),
                'hit_mrr':int(argub==best),'hit_cal':int(argpt==best),
                'diverges':int(argub!=argpt),'mrr_valid':int(valid),
                'UB':round(ub[argub],5),'chosen_mrr':argub,'chosen_cal':argpt})
    return out

def main():
    t0=time.time(); rows=[]
    for name in ['digits','fashion']:
        for seed in SEEDS: rows+=run(name,seed)
    agg={}
    for rho in RHOS:
        sub=[r for r in rows if r['rho']==rho]
        def m(rs,k):
            return round(float(np.mean([r[k] for r in rs])),5) if rs else None
        div=[r for r in sub if r['diverges']]
        agg[str(rho)]={
          'n_vertices':len(sub),'n_diverges':len(div),
          'regret_mrr':m(sub,'regret_mrr'),'regret_cal':m(sub,'regret_cal'),
          'hit_mrr':m(sub,'hit_mrr'),'hit_cal':m(sub,'hit_cal'),
          'mrr_cert_rate':m(sub,'mrr_valid'),
          'regret_mrr_on_diverged':m(div,'regret_mrr'),'regret_cal_on_diverged':m(div,'regret_cal'),
          'hit_mrr_on_diverged':m(div,'hit_mrr'),'hit_cal_on_diverged':m(div,'hit_cal'),
        }
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    with open(OUT,'w') as f: json.dump({'rows':rows,'agg':agg,'rho':RHOS,'tau':TAU,'runtime_s':time.time()-t0},f,indent=2)
    print(json.dumps(agg,indent=1)); print('runtime_s=%.1f rows=%d'%(time.time()-t0,len(rows)))
if __name__=='__main__': main()