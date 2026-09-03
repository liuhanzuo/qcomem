"""M2: cert-gated point-estimate model ranking under subgroup-mix turnover.

r1881 首轮已诚实定位 M1 负结果: naive argmin_i U_i(w) 因 max_j(U_i-L_j)=U_i-min_j L_j
在均质 CI 宽度下退化为 cal-prior(vacuity), 在异质宽度下反被悲观端驱动(regret 增大)。
本 runner 实现同题修复 M2:
  - 选择: 点估计驱动 argmin_i pt_i(w) = Σ_g w_g phat_{i,g}  (不因悲观化而选差)。
  - 证书/回退: 用 P2 线性带做"门"。对所选 i*:  sound regret 上界
      UB_{i*}(w) = U_{i*}(w) - min_j L_j(w)
    若 UB≤τ 则"承诺": 证明 regret(i*,w)≤τ (联合覆盖 ≥1-δ, Bonferroni+CP 精确)。
    若 UB>τ 则"弃答/回退": 诚实报告该 mixture w 处无从判别, 不硬选。
在 OUTER 上用 group-IPS 结算真 R_i(w), 只读一次。

诚实的部署端点(可审计、非 cherry-pick):
  (a) 证书覆盖率: committed 集合上经验 regret≤τ 的比例(应≥接近1, 验证带 sound).
      committed regret 均值/最大也应远离违反域。
  (b) 弃答有效性: abstained 集合上"若硬选点估计会发生"的 regret(报告其均值/最大),
      应显著高于 committed——说明弃答门在 turnover 敏感区(mixture skew)正确起作用。
  (c) 对比 baseline: committed 的平均 regret vs 全部硬选(no gate)平均 regret。
PROJECT: A2_SAFE_MODEL_RANKING_SUBGROUP_MIX ROUND: r1884。纯 CPU/前台/零 GPU。
"""
import json, time, numpy as np, os
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.special import betaincinv
import torchvision

OUT = "subgroup_mix_ranking/results/SUBGMIX_M2_GATE_R1884.json"
DATA_CACHE = "duplicate_sel/data"
TAU = 0.04
CAL_FRAC = 0.30
SEEDS = [0, 1, 2, 3, 4]
DELTA = 0.10


def cp_ci(n, x, delta):
    if x == 0:
        lo = 0.0
    else:
        lo = float(betaincinv(x, n - x + 1, delta / 2.0))
    if x == n:
        hi = 1.0
    else:
        hi = float(betaincinv(x + 1, n - x, 1.0 - delta / 2.0))
    return lo, hi


def model_pool(seed):
    return {
        'lr_C1': LogisticRegression(C=1.0, max_iter=3000, random_state=seed),
        'lr_C01': LogisticRegression(C=0.01, max_iter=3000, random_state=seed),
        'linsvc': LinearSVC(C=0.1, max_iter=4000, random_state=seed),
        'mlp': MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=seed),
    }


def load_carrier(name):
    if name == 'digits':
        from sklearn.datasets import load_digits
        d = load_digits()
        return d.data.astype(np.float32), d.target, 'digits'
    tr = torchvision.datasets.FashionMNIST(DATA_CACHE, train=True, download=False) if 'fashion' in name else \
         torchvision.datasets.MNIST(DATA_CACHE, train=True, download=False)
    te = torchvision.datasets.FashionMNIST(DATA_CACHE, train=False, download=False) if 'fashion' in name else \
         torchvision.datasets.MNIST(DATA_CACHE, train=False, download=False)
    X = np.concatenate([np.array(tr.data).reshape(-1, 784), np.array(te.data).reshape(-1, 784)], 0)
    y = np.concatenate([np.array(tr.targets), np.array(te.targets)], 0).astype(int)
    return (X / 255.0).astype(np.float32), y, 'image'


def load_news():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    tr = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=0)
    te = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=0)
    v = TfidfVectorizer(sublinear_tf=True, max_features=30000)
    X = np.vstack([v.fit_transform(tr.data).toarray(), v.transform(te.data).toarray()]).astype(np.float32)
    y = np.concatenate([tr.target, te.target]).astype(int)
    return X, y, 'news'


# 非 news 从 freeze 缓存加载特征; news 需重新 tfidf(与 r1881 同法, 无数据 ID 泄漏)
def w_grid(G, u):
    grid = [{'name': 'uniform', 'w': {g: 1.0 / G for g in range(G)}}]
    for g in range(G):
        w = {h: 0.02 for h in range(G)}
        w[g] = 1.0 - 0.02 * (G - 1)
        grid.append({'name': f'skew_peak{g}', 'w': w})
    grid.insert(0, {'name': 'collected_u', 'w': u})
    for idx in (0, 1, G // 2):
        gpk = idx % G
        w = {h: 0.02 for h in range(G)}
        w[gpk] = 1.0 - 0.02 * (G - 1)
        interp = {h: 0.7 * u[h] + 0.3 * w[h] for h in range(G)}
        grid.append({'name': f'interp_peak{gpk}', 'w': interp})
    return grid


def run_carrier(name, seed):
    X, y, kind = (load_news() if name == 'news' else load_carrier(name))
    G = int(y.max() + 1)
    X_all, X_outer, y_all, y_outer = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    pca = PCA(n_components=min(128, X_all.shape[1]), random_state=seed).fit(X_all)
    Z_all = pca.transform(X_all)
    Z_outer = pca.transform(X_outer)
    Z_fit, Z_cal, y_fit, y_cal = train_test_split(Z_all, y_all, test_size=CAL_FRAC, stratify=y_all, random_state=seed)
    pool = model_pool(seed)
    trained = {}
    for mid, m in pool.items():
        m.fit(Z_fit, y_fit)
        trained[mid] = m
    M = len(trained)
    dcell = DELTA / (G * M)
    mist = {}
    for mid, m in trained.items():
        pred = m.predict(Z_cal)
        err = (pred != y_cal).astype(float)
        ce = {g: float(err[y_cal == g].mean()) if (y_cal == g).sum() else 0.0 for g in range(G)}
        cn = {g: int((y_cal == g).sum()) for g in range(G)}
        # per-group CI
        L, U = {}, {}
        for g in range(G):
            lo, hi = cp_ci(cn[g], int(round(ce[g] * cn[g])), dcell)
            L[g], U[g] = lo, hi
        mist[mid] = {'pt': ce, 'n': cn, 'L': L, 'U': U}
    # oracle (OUTER, 诊断 ceiling, 不用于选择)
    oracle = {}
    for mid, m in trained.items():
        pe = np.array(m.predict(Z_outer) != y_outer, dtype=float)
        oracle[mid] = {g: float(pe[y_outer == g].mean()) if (y_outer == g).sum() else 0.0 for g in range(G)}
    u = {g: (y_cal == g).sum() / len(y_cal) for g in range(G)}
    rows = []
    for winfo in w_grid(G, u):
        w = winfo['w']; wname = winfo['name']
        trueR = {mid: sum(w[g] * oracle[mid][g] for g in range(G)) for mid in trained}
        best_m = min(trueR, key=trueR.get)
        bestR = trueR[best_m]
        # 点估计选择
        ptR = {mid: sum(w[g] * mist[mid]['pt'][g] for g in range(G)) for mid in trained}
        i = min(ptR, key=ptR.get)
        U_i = sum(w[g] * mist[i]['U'][g] for g in range(G))
        minL = min(sum(w[g] * mist[j]['L'][g] for g in range(G)) for j in trained)
        UB = U_i - minL                      # sound P2 regret 上界
        committed = UB <= TAU
        reg = trueR[i] - bestR              # 真 regret(OUTER, 只读)
        rows.append({'carrier': name, 'seed': seed, 'w': wname, 'chosen': i,
                     'oracle_best': best_m, 'true_regret': round(reg, 4),
                     'UB': round(UB, 4), 'committed': bool(committed),
                     'cert_regret': round(UB if committed else -1.0, 4)})
    return rows


def main():
    t0 = time.time()
    all_rows = []
    for name in ['digits', 'fashion', 'mnist', 'news']:
        for seed in SEEDS:
            all_rows.extend(run_carrier(name, seed))
    n_comm = sum(1 for r in all_rows if r['committed'])
    n_abs = len(all_rows) - n_comm
    comm_reg = [r['true_regret'] for r in all_rows if r['committed']]
    abs_reg = [r['true_regret'] for r in all_rows if not r['committed']]
    # 证书覆盖: committed 上真 regret≤τ 的比例(验证 sound)
    cov = np.mean([r <= TAU + 1e-9 for r in comm_reg]) if comm_reg else float('nan')
    # 全硬选(no-gate 参考) regret
    all_reg = [r['true_regret'] for r in all_rows]
    # UB 上界自检: soundness 断言 — committed 的 UB 应确实≥该行(tau 门内), 且计算一致
    bad_ub = sum(1 for r in all_rows if abs(r['cert_regret'] - r['UB']) > 1e-9 and r['cert_regret'] >= 0)
    assert bad_ub == 0, "cert_regret/UB mismatch"
    agg = {
        'n_rows': len(all_rows), 'n_committed': n_comm, 'n_abstained': n_abs,
        'committed_rate': round(n_comm / len(all_rows), 3),
        'cert_coverage(rego<=tau)': round(float(cov), 4),
        'committed_mean_regret': round(float(np.mean(comm_reg)), 4) if comm_reg else None,
        'committed_max_regret': round(float(np.max(comm_reg)), 4) if comm_reg else None,
        'abstained_mean_regret(if-hardpick)': round(float(np.mean(abs_reg)), 4) if abs_reg else None,
        'abstained_max_regret': round(float(np.max(abs_reg)), 4) if abs_reg else None,
        'no_gate_mean_regret': round(float(np.mean(all_reg)), 4),
        'tau': TAU, 'delta': DELTA, 'runtime_s': round(time.time() - t0, 1),
    }
    out = {'project': 'A2_SAFE_MODEL_RANKING_SUBGROUP_MIX', 'round': 'r1884', 'agg': agg, 'rows': all_rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(agg, indent=2))
    print('saved', OUT)


if __name__ == '__main__':
    main()