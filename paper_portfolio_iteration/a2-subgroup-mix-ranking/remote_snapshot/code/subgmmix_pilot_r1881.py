"""FINITE-SAMPLE SAFE MODEL RANKING UNDER SUBGROUP-MIX TURNOVER — 首个真实 readback pilot.

PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX
ROUND: orchestrator 1881 / A2 work-tag r1881. 纯 CPU/前台/零 GPU.

设计（见 PROJECT_DESIGN.md）:
- carrier: digits, Fashion-MNIST, MNIST (PCA128 冻结特征), subgroup = class.
- FIT: 训固定轻量候选模型 {LR-C, LR-C01, LinearSVC, MLP} (在 CAL 上打分 risk-est.)
- CAL: 估每模型每 ground-truth 组的 0-1 错误率 + Clopper-Pearson 同时 CI (Bonferroni)。
- W: 预声明未来 mixture 网格 (Gx side)。用 label-shift IPS 重加权在 OUTER 上估计真 R_i(w)。
- 规则:
    * oracle (不可部署, 用于诊断 CEILING): 用 OUTER 真逐组错误率选 argmin Σw r。
    * cal-prior (点估计: 强 baseline): argmin Σw p̂_{i,g} (利用 CAL 逐组错误率点估计)。
    * mrr (本文): argmin_i UB_i(w) = max_j (U_i−L_j), 并给 regret≤UB 证书; 若全 UB>τ 回退。
    * worst-group: argmin_i max_g r_{i,g} (与 w 无关)。
    * equal-mixture: w=均匀 (与 w 无关, 相对只看采集 w 是另一种稳健先验)。
- EVAL 一次只读结算: R_i(w)=IPS(Σ_g w_g · 1[hat!=y] / P(Y=g|X)) 在封存 OUTER 上。

端点: 未来-mixture 平均 regret、rank reversal、选到 oracle 最优的命中率、证书/回退率、
带宽 + 成本。
诚实边界: 不使用数据集 ID/事后 oracle 特征做选择; EVAL 严格 outsample。
"""
import json, time, numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits, fetch_20newsgroups
import torchvision  # Fashion/MNIST 冻结特征
import torch

DIGITS_CLASSES = 10
FASH_CLASSES = 10
OUT_ROOT = "subgroup_mix_ranking/results"
SEEDS = [0, 1, 2, 3, 4]
TAU = 0.06          # 证书阈值 (regret 上界容忍)
CAL_FRAC = 0.30     # 用一部分训练做 CAL (估逐组错误率), 余下做 FIT
N_CAL_MIN = 1200

def _clopper_pearson(n, x, delta):
    """Clopper-Pearson exact (upper) CI for Bernoulli p given n trials, x success.
    Return (lo, hi) for Binomial. delta = 1-ci."""
    from scipy.special import betaincinv
    if x == 0:
        lo = 0.0
    else:
        lo = betaincinv(x, n - x + 1, delta / 2.0)
    if x == n:
        hi = 1.0
    else:
        hi = betaincinv(x + 1, n - x, 1.0 - delta / 2.0)
    return float(lo), float(hi)

def w_errmat(model, X, y_groups):
    """Return per-(model,group) misclassification {R: g->err} and count over CAL/X,
    X already in frozen-feature space. Computes error on the passed X (outer for oracle)."""
    pred = model.predict(X)
    err = (pred != y_groups).astype(float)
    sums, cnts, errs = {}, {}, {}
    for g in np.unique(y_groups):
        m = y_groups == g
        sums[g] = err[m].sum(); cnts[g] = int(m.sum())
        errs[g] = sums[g] / cnts[g] if cnts[g] else 0.0
    return errs, cnts

def calc_ci(errs, cnts, delta):
    L, U = {}, {}
    for g in errs:
        if cnts[g] == 0:
            L[g], U[g] = 0.0, 0.0; continue
        lo, hi = _clopper_pearson(cnts[g], int(round(errs[g] * cnts[g])), delta)
        L[g], U[g] = lo, hi
    return L, U

def linear_band(w, L, U):
    return sum(w[g] * L[g] for g in w), sum(w[g] * U[g] for g in w)

def make_pca(X, k=128, seed=0):
    return PCA(n_components=k, random_state=seed).fit(X)

def _load_digits():
    from sklearn.datasets import load_digits as _ld
    d = _ld()
    return d.data.astype(np.float32), d.target, d.data.shape[1]

DATA_CACHE = "duplicate_sel/data"

def _load_fashion():
    tr = torchvision.datasets.FashionMNIST(DATA_CACHE, train=True, download=False)
    te = torchvision.datasets.FashionMNIST(DATA_CACHE, train=False, download=False)
    X = np.concatenate([np.array(tr.data).reshape(-1, 784), np.array(te.data).reshape(-1, 784)], 0)
    y = np.concatenate([np.array(tr.targets), np.array(te.targets)], 0).astype(int)
    return X.astype(np.float32) / 255.0, y, 784

def _load_mnist():
    tr = torchvision.datasets.MNIST(DATA_CACHE, train=True, download=False)
    te = torchvision.datasets.MNIST(DATA_CACHE, train=False, download=False)
    X = np.concatenate([np.array(tr.data).reshape(-1, 784), np.array(te.data).reshape(-1, 784)], 0)
    y = np.concatenate([np.array(tr.targets), np.array(te.targets)], 0).astype(int)
    return X.astype(np.float32) / 255.0, y, 784

def _load_20news():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    cat = [c for c in range(20)]
    tr = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'), shuffle=True, random_state=0)
    te = fetch_20newsgroups(subset='test',  remove=('headers','footers','quotes'), shuffle=True, random_state=0)
    # use target as group (20 classes) — 预处理 tfidf->PCA128 frozen
    v = TfidfVectorizer(sublinear_tf=True, max_features=30000)
    Xtr_tf = v.fit_transform(tr.data); Xte_tf = v.transform(te.data)
    X = np.vstack([Xtr_tf.toarray(), Xte_tf.toarray()]).astype(np.float32)
    y = np.concatenate([tr.target, te.target]).astype(int)
    return X, y, X.shape[1]

DATA_LOADERS = {'digits': _load_digits, 'fashion': _load_fashion,
                'mnist': _load_mnist, 'news': _load_20news}

def build_models(n_classes, n_dim, seed=0):
    """固定候选模型族。全部在冻结特征上训练, 用 CAL 组风险做选择。"""
    C = 0.1
    return {
        'lr_C1':   LogisticRegression(C=1.0, max_iter=3000, random_state=seed),
        'lr_C01':  LogisticRegression(C=0.01, max_iter=3000, random_state=seed),
        'linsvc':  LinearSVC(C=0.1, max_iter=4000, random_state=seed),
        'mlp':     MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=seed),
    }

def w_grid(G, seeds):
    """预声明未来 mixture 网格: label-skew 尖峰 + 均匀 + 与采集 u 插值(在子函数做)。"""
    grid = []
    # 均匀
    grid.append({'name': 'uniform', 'w': {g: 1.0/G for g in range(G)}})
    # 单类尖峰
    for g in range(G):
        w = {h: 0.02 for h in range(G)}
        w[g] = 1.0 - 0.02*(G-1)
        grid.append({'name': f'skew_peak{g}', 'w': w})
    return grid

def ips_risk(w, y_groups, err_mask, class_prob, G):
    """IPS / label-shift estimator of R(w)=Σ_g w_g P(err|g) using OUTER class-cond prob.
    err_mask: 1 where model misclassifies on OUTER. returns weighted estimate."""
    # 用组条件错误率的直接估计 + 组先验 w_g (组内无 covariate shift 假设)
    s = 0.0
    for g in range(G):
        m = y_groups == g
        if m.sum():
            s += w[g] * err_mask[m].mean()
    return s

def run_carrier(name, seed, rng):
    X, y, ndim = DATA_LOADERS[name]()
    G = int(y.max() + 1)
    # --- split: OUTER(封存, EVAL) | 其余(fit+cal) ---
    X_all, X_outer, y_all, y_outer = train_test_split(X, y, test_size=0.25,
                                                      stratify=y, random_state=seed)
    # --- frozen feature: PCA fit on (all \ outer) only, no data-ID leakage to outer ---
    pca = make_pca(X_all, k=min(128, X_all.shape[1]), seed=seed)
    Z_all = pca.transform(X_all) if pca.n_components_ else X_all
    Z_outer = pca.transform(X_outer) if pca.n_components_ else X_outer
    # --- further split all into fit / cal ---
    Z_fit, Z_cal, y_fit, y_cal = train_test_split(Z_all, y_all, test_size=CAL_FRAC,
                                                   stratify=y_all, random_state=seed)
    models = build_models(G, Z_fit.shape[1], seed=seed)
    # 训练候选 (FIT set)
    trained = {}
    for mid, m in models.items():
        m.fit(Z_fit, y_fit)
        trained[mid] = m
    # 全局 Bonferroni: 每 (model, group) 给 δ_cell = delta/(G*M)  每个 O(CI) 并行
    M = len(trained)
    delta = 0.10
    dcell = delta / (G * M)
    # 校准: 每模型每组的点估计 + 同时 CI (在 CAL set, group = true label)
    mist = {}
    for mid, m in trained.items():
        pred = m.predict(Z_cal)
        err = (pred != y_cal).astype(float)
        ce = {g: float(err[y_cal == g].mean()) if (y_cal == g).sum() else 0.0
              for g in range(G)}
        cn = {g: int((y_cal == g).sum()) for g in range(G)}
        L, U = calc_ci(ce, cn, dcell)
        mist[mid] = {'pt': ce, 'count': cn, 'L': L, 'U': U}
    # oracle 逐组风险 (OUTER, 仅诊断 ceiling) —— 严格 outsample, 不用于部署选择
    oracle = {}
    for mid, m in trained.items():
        pe = np.array(m.predict(Z_outer) != y_outer, dtype=float)
        oe = {g: float(pe[y_outer == g].mean()) if (y_outer == g).sum() else 0.0
              for g in range(G)}
        oracle[mid] = oe
    # --- pre-declared W grid ---
    ww = w_grid(G, seed)
    # 采集 mixture u: 用 CAL 的组频 (可观察), 作为"当前域"一个点
    u = {g: (y_cal == g).sum() / len(y_cal) for g in range(G)}
    ww.insert(0, {'name': 'collected_u', 'w': u})
    # 插值 toward skew peak: 再加 2 个插值点
    for idx in (0, 1, G // 2):
        gpk = idx % G if idx < G else (idx % G)
        w = {h: 0.02 for h in range(G)}; w[gpk] = 1.0 - 0.02*(G-1)
        interp = {h: 0.7*u[h] + 0.3*w[h] for h in range(G)}
        ww.append({'name': f'interp_peak{gpk}', 'w': interp})
    # ---- evaluate all rules on W over OUTER ----
    rows = []
    for winfo in ww:
        w = winfo['w']; Wname = winfo['name']
        true_R = {}
        for mid, oe in oracle.items():
            true_R[mid] = sum(w[g] * oe[g] for g in range(G))
        best_m = min(true_R, key=true_R.get)
        bestR = true_R[best_m]
        # oracle 最优
        row = {'w': Wname, 'oracle_best': best_m, 'oracle_R': round(bestR, 4)}
        # 决策规则集合
        decs = {}
        # cal-prior: argmin Σw pt
        cp_pt = {mid: sum(w[g] * mist[mid]['pt'][g] for g in range(G)) for mid in trained}
        decs['cal_prior'] = min(cp_pt, key=cp_pt.get)
        # equal-mixture
        weq = {g: 1.0/G for g in range(G)}
        eq_pt = {mid: sum(weq[g] * mist[mid]['pt'][g] for g in range(G)) for mid in trained}
        decs['equal_mixture'] = min(eq_pt, key=eq_pt.get)
        # worst-group (与 w 无关, 用 point 上界 e.g. max_g pt)
        wg = {mid: max(mist[mid]['pt'][g] for g in range(G)) for mid in trained}
        decs['worst_group'] = min(wg, key=wg.get)
        # mrr: 用线性带 minimax-regret
        ub = {}
        for mid in trained:
            U = mist[mid]['U']
            U_i = sum(w[g] * U[g] for g in range(G))
            worst_j = 0.0
            for m2 in trained:
                if m2 == mid: continue
                L_j = sum(w[g] * mist[m2]['L'][g] for g in range(G))
                worst_j = max(worst_j, U_i - L_j)
            ub[mid] = worst_j
        best_ub = min(ub, key=ub.get)
        mmr_valid = ub[best_ub] <= TAU
        decs['mrr'] = best_ub
        decs['mrr_valid'] = mmr_valid
        # 回退: mrr 全分支 UB>TAU => fall back to cal_prior (稳健混合/abstain 代理)
        if not mmr_valid:
            decs['mrr_valid'] = False
            decs['mrr_fallback'] = True
        for rule, chosen in list(decs.items()):
            if rule in ('mrr_valid', 'mrr_fallback'): continue
            if rule == 'mrr':
                chosen = best_ub if mmr_valid else decs['cal_prior']
            row[f'{rule}_chosen'] = chosen
            row[f'{rule}_regret'] = round(true_R[chosen] - bestR, 4)
            row[f'{rule}_hit'] = 1 if chosen == best_m else 0
        rows.append(row)
    return {'carrier': name, 'seed': seed, 'G': G, 'M': M, 'delta': delta,
            'rows': rows, 'cal_counts': mist['lr_C1']['count'],
            'cal_pt_lrC1': mist['lr_C1']['pt']}

def main():
    t0 = time.time()
    out = {'projects': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1881',
           'tau': TAU, 'cal_frac': CAL_FRAC, 'carriers': {}, 'global': {}}
    all_reg = {'mrr': [], 'cal_prior': [], 'equal_mixture': [], 'worst_group': []}
    all_hit = {'mrr': [], 'cal_prior': [], 'equal_mixture': [], 'worst_group': []}
    # 载体间结果聚合 (跨 carrier+seed+wide W-grid)
    for name in ['digits', 'fashion', 'mnist', 'news']:
        crows = []
        for seed in SEEDS:
            r = run_carrier(name, seed, np.random.RandomState(seed))
            out['carriers'].setdefault(name, {'reps': []})['reps'].append(r)
            for row in r['rows']:
                for rule in ['mrr', 'cal_prior', 'equal_mixture', 'worst_group']:
                    all_reg[rule].append(row[f'{rule}_regret'])
                    all_hit[rule].append(row[f'{rule}_hit'])
            crows.append(r)
    agg = {}
    for rule in ['mrr', 'cal_prior', 'equal_mixture', 'worst_group']:
        agg[rule] = {'mean_regret': round(float(np.mean(all_reg[rule])), 4),
                     'mean_abs_regret': round(float(np.mean(np.abs(all_reg[rule]))), 4),
                     'hit_rate': round(float(np.mean(all_hit[rule])), 4),
                     'n': len(all_reg[rule])}
    # 证书/回退率 (跨全部 W grid rows, mrr 分支)
    n_mrr, n_valid, n_fb = 0, 0, 0
    for name in SEEDS and ['digits']:
        pass
    val_cert = []
    for name in ['digits', 'fashion', 'mnist', 'news']:
        for rep in out['carriers'][name]['reps']:
            for row in rep['rows']:
                n_mrr += 1
                if row.get('mrr_fallback', False):
                    n_fb += 1
                else:
                    n_valid += 1
    agg['mrr'] = {**agg['mrr'],
                  'cert_rate': round(n_valid / n_mrr, 3), 'fallback_rate': round(n_fb / n_mrr, 3)}
    out['global'] = agg
    out['n_rows'] = n_mrr
    out['runtime_s'] = time.time() - t0
    import os
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, 'SUBGMIX_PILOT_R1881.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'agg': agg, 'n_rows': n_mrr, 'runtime_s': round(out['runtime_s'],1),
                      'path': path}, indent=2))
    # 打印一两个载体的代表性 w 网格 row 前几行, 便于人工读
    for name in ['digits', 'news']:
        rep = out['carriers'][name]['reps'][0]
        print(f"\n[{name} seed0 sample rows]")
        for row in rep['rows'][:3]:
            print({k: v for k, v in row.items() if not isinstance(v, dict)})

if __name__ == '__main__':
    main()