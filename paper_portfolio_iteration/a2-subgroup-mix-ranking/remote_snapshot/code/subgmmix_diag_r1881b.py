"""r1881b 聚焦诊断: 揭示 mrr(cert)规则为何与 cal_prior 相同, 并让证书可判别.

从 SUBGMIX_PILOT_R1881.json 读数据不可行(未存逐组数量/上界)。本文件只跑 digits+fashion
(轻), 多存: 每模型每 w 的 [点估计, L, U, UB_i, argmin_U vs argmin_pt 是否分歧],
TAU 扫描 → 证书率/回退率/证书区内 regret(MMR vs cal_prior)的判别。同题修复, 不改题。
"""
import json, time, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.special import betaincinv
import torchvision
import os

OUT = "subgroup_mix_ranking/results/SUBGMIX_DIAG_R1881B.json"
TAUS = [0.03, 0.05, 0.08, 0.12, 0.18]
DATA_CACHE = "duplicate_sel/data"

def cp_ci(n, x, delta):
    if x == 0: lo = 0.0
    else: lo = betaincinv(x, n-x+1, delta/2.0)
    if x == n: hi = 1.0
    else: hi = betaincinv(x+1, n-x, 1.0-delta/2.0)
    return float(lo), float(hi)

def _load_digits():
    from sklearn.datasets import load_digits as _ld
    d = _ld(); return d.data.astype(np.float32), d.target

def _load_fashion():
    tr = torchvision.datasets.FashionMNIST(DATA_CACHE, train=True, download=False)
    te = torchvision.datasets.FashionMNIST(DATA_CACHE, train=False, download=False)
    X = np.concatenate([np.array(tr.data).reshape(-1,784), np.array(te.data).reshape(-1,784)],0)
    y = np.concatenate([np.array(tr.targets), np.array(te.targets)],0).astype(int)
    return X.astype(np.float32)/255.0, y

def build(nc, nd, seed):
    return {
      'lr_C1': LogisticRegression(C=1.0, max_iter=3000, random_state=seed),
      'lr_C01': LogisticRegression(C=0.01, max_iter=3000, random_state=seed),
      'linsvc': LinearSVC(C=0.1, max_iter=4000, random_state=seed),
      'mlp': MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=seed),
    }

def run(name, seed):
    rng = np.random.RandomState(seed)
    X, y = _load_digits() if name=='digits' else _load_fashion()
    G = int(y.max()+1)
    pca = PCA(n_components=min(128, X.shape[1]), random_state=seed).fit(X)
    Z = pca.transform(X) if pca.n_components_ else X
    # OUTER 25% / rest
    Xr, Xo, yr, yo = train_test_split(Z, y, test_size=0.25, stratify=y, random_state=seed)
    Zf, Zc, yf, yc = train_test_split(Xr, yr, test_size=0.30, stratify=yr, random_state=seed)
    mods = build(G, Zf.shape[1], seed)
    tr = {}
    for k,m in mods.items(): m.fit(Zf, yf); tr[k]=m
    M=len(tr); delta=0.10; dcell=delta/(G*M)
    mist={}
    for k,m in tr.items():
        p=m.predict(Zc); err=(p!=yc).astype(float)
        ce={g:float(err[yc==g].mean()) if (yc==g).sum() else 0.0 for g in range(G)}
        cn={g:int((yc==g).sum()) for g in range(G)}
        L,U=cp_ci(cn, {g:int(round(ce[g]*cn[g])) for g in range(G)}, dcell) if False else ({g:0 for g in range(G)},{g:0 for g in range(G)})
        Ls,Us={},{}
        for g in range(G):
            lo,hi=cp_ci(cn[g], int(round(ce[g]*cn[g])), dcell); Ls[g],Us[g]=lo,hi
        mist[k]={'pt':ce,'n':cn,'L':Ls,'U':Us}
    orc={}
    pe_o={}
    for k,m in tr.items():
        p=m.predict(Xo); em=(p!=yo).astype(float); pe_o[k]=em
        orc[k]={g:float(em[yo==g].mean()) if (yo==g).sum() else 0.0 for g in range(G)}
    u={g:(yc==g).sum()/len(yc) for g in range(G)}
    wgrid=[{'name':'u','w':u}]
    wgrid.append({'name':'uniform','w':{g:1.0/G for g in range(G)}})
    for g in range(G):
        w={h:0.02 for h in range(G)}; w[g]=1.0-0.02*(G-1)
        wgrid.append({'name':f'skew{g}','w':w})
        wgrid.append({'name':f'interp{g}','w':{h:0.7*u[h]+0.3*w[h] for h in range(G)}})
    out_rows=[]
    all_res={k:[] for k in ['mrr','cal_prior','equal_mixture','worst_group']}
    for wi in wgrid:
        w=wi['w']; wnm=wi['name']
        tr_R={k:sum(w[g]*orc[k][g] for g in range(G)) for k in tr}
        best=min(tr_R,key=tr_R.get); bestR=tr_R[best]
        # 候选序: U_i 与 pt_i 是否不同 argmin
        pt_i={k:sum(w[g]*mist[k]['pt'][g] for g in range(G)) for k in tr}
        U_i={k:sum(w[g]*mist[k]['U'][g] for g in range(G)) for k in tr}
        L_bar=min({k:sum(w[g]*mist[k]['L'][g] for g in range(G)) for k in tr}.values())
        ub={k:U_i[k]-L_bar for k in tr}
        argpt=min(pt_i,key=pt_i.get); argU=min(U_i,key=U_i.get); argub=min(ub,key=ub.get)
        diverges = (argU != argpt) or (argub != argpt)
        # mrr chosen = argmin ub (=argmin U) 若有效, 否则回退 cal_prior(argpt)
        wg={k:max(mist[k]['pt'][g] for g in range(G)) for k in tr}
        eqpt={k:sum((1.0/G)*mist[k]['pt'][g] for g in range(G)) for k in tr}
        for tau in TAUS:
            valid = ub[argub] <= tau
            mrrc = argub if valid else argpt
            row={'carrier':name,'seed':seed,'w':wnm,'tau':tau,
                 'oracle_best':best,'oracle_R':round(bestR,5),
                 'diverges_U': int(argU!=argpt), 'diverges_UB': int(argub!=argpt),
                 'argU':argU,'argpt':argpt,'UB_argub':round(ub[argub],5),
                 'regret_mrr':round(tr_R[mrrc]-bestR,5),
                 'regret_cal':round(tr_R[argpt]-bestR,5),
                 'regret_eq':round(tr_R[min(eqpt,key=eqpt.get)]-bestR,5),
                 'regret_wg':round(tr_R[min(wg,key=wg.get)]-bestR,5),
                 'hit_mrr':int(mrrc==best),'hit_cal':int(argpt==best),
                 'mrr_valid':int(valid),'mrr_fallback':int(not valid)}
            out_rows.append(row)
    return out_rows

def main():
    t0=time.time(); rows=[]
    for name in ['digits','fashion']:
        for seed in [0,1,2,3]:
            rows += run(name,seed)
    # 聚合: 每种证书区(按 tau) mrr vs cal vs eq vs wg 的 mean regret & hit
    agg={}
    for tau in TAUS:
        sub=[r for r in rows if r['tau']==tau]
        cert=[r for r in sub if r['mrr_valid']]
        fb=[r for r in sub if not r['mrr_valid']]
        def stat(rs,key):
            if not rs: return None
            vals=[r[key] for r in rs]
            return round(float(np.mean(vals)),5)
        agg[str(tau)]={
          'n':len(sub),'cert_n':len(cert),'fb_n':len(fb),
          'cert_rate':round(len(cert)/len(sub),3),
          'cert_regret_mrr':stat(cert,'regret_mrr'),'cert_regret_cal':stat(cert,'regret_cal'),
          'cert_regret_eq':stat(cert,'regret_eq'),'cert_regret_wg':stat(cert,'regret_wg'),
          'cert_hit_mrr':stat(cert,'hit_mrr'),'cert_hit_cal':stat(cert,'hit_cal'),
          'all_regret_mrr':stat(sub,'regret_mrr'),'all_regret_cal':stat(sub,'regret_cal'),
          'all_hit_mrr':stat(sub,'hit_mrr'),'all_hit_cal':stat(sub,'hit_cal'),
          'diverges_U_frac': round(sum(r['diverges_U'] for r in sub)/len(sub),3),
          'diverges_UB_frac': round(sum(r['diverges_UB'] for r in sub)/len(sub),3),
        }
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    with open(OUT,'w') as f: json.dump({'rows':rows,'agg':agg,'runtime_s':time.time()-t0},f,indent=2)
    print(json.dumps(agg,indent=1))
    print('runtime_s=%.1f'%(time.time()-t0))

if __name__=='__main__': main()