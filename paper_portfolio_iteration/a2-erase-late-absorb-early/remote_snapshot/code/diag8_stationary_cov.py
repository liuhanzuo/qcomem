"""diag8 (r780): T3 noise-floor lemma numeric lock.
Measure the SGD stationary covariance around the clean attractor w*_A in the
low-LR tail, and test the prediction  Tr(Cov) ~ eta^2 * K * sigma^2 / mu.

Setup: same convex data as pilot14 (make_data), clean training only (window
None). After burn-in (full schedule), keep SGD at constant eta=0.08 for M
extra epochs and collect the endpoint sample; repeat over 16 independent SGD
runs (different batch-permutation RNG, same data seed and same init w0) to
estimate Cov of the stationary endpoint distribution.

Predictions (T3 lemma):
  V1 iso-scaling: Tr(Cov) / (eta^2 * K) ~ constant across bs (sigma^2/mu fixed).
  V2 monotone: Tr(Cov) increases as bs decreases (K increases).
Report also per-bs endpoint distance to w*_A (GD clean endpoint) — this is the
noise-floor radius relevant to diag7's "w stays on damage attractor".
"""
import numpy as np, json, time, sys
sys.path.insert(0, '.')
from pilot7_ablate2x2 import sigmoid
from pilot5_boundary import make_data

t0 = time.time()
lam = 1e-3
T, T_split = 240, 120
ETA = np.array([0.8 if t < T_split else 0.08 for t in range(T)])
ETA_LO = 0.08
M_EXTRA = 30          # extra constant-LR epochs to reach stationary
N_RUNS = 16           # independent SGD endpoint samples
d = 40

def grad_full(X, y, w):
    z = X @ w; p = sigmoid(z)
    return X.T @ (p - y) / len(y) + lam * w

def sgd_endpoint(XA, yA, w0, bs, run):
    rng = np.random.default_rng(50_000 + run)
    w = w0.copy()
    # burn-in: full schedule
    for t in range(T):
        eta = ETA[t]
        perm = rng.permutation(len(yA))
        for i in range(0, len(yA), bs):
            idx = perm[i:i + bs]
            z = XA[idx] @ w; p = sigmoid(z)
            g = XA[idx].T @ (p - yA[idx]) / len(idx) + lam * w
            w = w - eta * g
    # stationary sampling at constant low LR
    for _ in range(M_EXTRA):
        perm = rng.permutation(len(yA))
        for i in range(0, len(yA), bs):
            idx = perm[i:i + bs]
            z = XA[idx] @ w; p = sigmoid(z)
            g = XA[idx].T @ (p - yA[idx]) / len(idx) + lam * w
            w = w - ETA_LO * g
    return w

# data + shared init (data seed 0, as in diag7)
XA, yA, XD, yD, Xt, yt = make_data(0)
w0 = np.random.default_rng(1000).standard_normal(d) * 0.05

# reference: deterministic GD clean attractor w*_A
w_star = w0.copy()
for t in range(T):
    w_star = w_star - ETA[t] * grad_full(XA, yA, w_star)
# polish GD attractor at low LR (same extra epochs, deterministic full batch)
for _ in range(M_EXTRA):
    w_star = w_star - ETA_LO * grad_full(XA, yA, w_star)

# single-batch full gradient at w* for sigma^2 estimate:
# sigma^2 := E||g_i - g_full||^2 with g_i the per-sample-loss gradient estimate
# used implicitly by mini-batch: g_batch = (1/bs) sum_i x_i (p_i - y_i) + lam w.
# We estimate the per-sample stochastic-gradient covariance at w*.
z = XA @ w_star; p = sigmoid(z)
r = (p - yA)                      # (n,) residual
G = XA * r[:, None]               # per-sample data-gradient (n,d), mean over i = data grad
g_full = G.mean(0) + lam * w_star
Gc = G - G.mean(0)
sigma2 = float((np.mean((Gc ** 2).sum(1))))   # E||g_i - gbar||^2 (per-sample scale)
# note: mini-batch gradient noise Var ~ sigma2/bs; accumulated per epoch over K
# steps gives the eta^2 K sigma2 scaling in Tr(Cov) (K*bs=n fixed => K*sigma2/bs
# ... we test the raw eta^2*K*sigma2/mu prediction directly).

rows = {}
for bs in [256, 128, 64, 32]:
    K = int(np.ceil(len(yA) / bs))
    W = np.stack([sgd_endpoint(XA, yA, w0, bs, r_) for r_ in range(N_RUNS)])
    mean_w = W.mean(0)
    C = np.cov((W - mean_w).T)
    tr = float(np.trace(C))
    dist_star = float(np.linalg.norm(mean_w - w_star))
    dist_runs_med = float(np.median(np.linalg.norm(W - w_star, axis=1)))
    rows[f'bs{bs}'] = dict(K=K, trace_cov=tr,
                           trace_over_eta2K=tr / (ETA_LO ** 2 * K),
                           dist_mean_to_wstar=dist_star,
                           dist_runs_to_wstar_med=dist_runs_med)

traces = np.array([rows[k]['trace_cov'] for k in ['bs256', 'bs128', 'bs64', 'bs32']])
scaled = np.array([rows[k]['trace_over_eta2K'] for k in ['bs256', 'bs128', 'bs64', 'bs32']])
V1_iso = float(scaled.max() / max(scaled.min(), 1e-12))
V2_monotone = bool(np.all(np.diff(traces) > 0))
mu_est = lam  # strong-convexity floor (ridge); logistic adds data curvature
out = dict(rows=rows, sigma2_persample=sigma2, mu_floor=mu_est,
           eta_lo=ETA_LO, M_extra=M_EXTRA, N_runs=N_RUNS,
           pred_trace_const_eta2Kmu=sigma2 / mu_est,
           V1_iso_ratio_max_over_min=V1_iso, V2_monotone_in_K=V2_monotone,
           runtime_sec=round(time.time() - t0, 1))
json.dump(out, open('diag8_stationary_cov_out.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
