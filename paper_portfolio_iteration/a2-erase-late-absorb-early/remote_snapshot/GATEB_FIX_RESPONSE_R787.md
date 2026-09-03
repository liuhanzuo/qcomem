# GATE-B 修复响应 r787（对 A3 T2/T3 主审 AUDIT_A2_GATEB_T23_R311 的 2 个 minor）

A3 判定通过(2 minor)，独立复算全部对上。本文件记录 F1/F2 修复落点（paper/paper.tex，
编译 0 错 0 undefined，9 页）。

## F1（diag5 两点隐含 λ_eff 差 1.8×，"定量吻合"过强）——已修
- A3 复核：s=30 点（E_tail=81.6, 收缩 0.0192）隐含 λ=ln(1/0.0192)/81.6=0.048；
  s=120 点（E_tail=9.6, 收缩 0.4297）隐含 λ=0.088——1.8× 差距。
- 我独立复算确认：两点实测收缩都**快于** λ_eff=0.037 保守预测（预测 0.049/0.701 vs
  实测 0.019/0.430）——λ_eff 下界成立（实测更好），但逐点定量精度不成立。
- 修复（§T2 mechanism 段）："matches quantitatively" 改为 "shrink faster than the
  conservative λ_eff=0.037 prediction (0.049/0.701); pointwise-implied rates 0.048/0.088
  (1.8× spread), so we claim the exponential law qualitatively with the λ_eff lower bound,
  not pointwise quantitative agreement — curvature grows along the contraction path
  (Lemma C1; same depth dependence as §T3)"。与 T3 honest-boundary 段的 λ(δ) 深度依赖
  呼应，统一口径。

## F2（T3 的 K 在 GD/SGD 口径不一致）——已修
- 修复（定理陈述）：K per-epoch 口径改为 **per-step 暴露预算** B:=Σ_steps η = N_win·η
  （等价 B=K·E_win），声明 GD/SGD 只在 window 含多少步上不同，B 已计入——消除
  (I-ηH)^K 同梯度幂与 SGD K=n/bs 不同 minibatch 的张力。
- 修复（proof skeleton step 2）：补 shuffle 鞅差条件一句——conditional on shuffle
  history, per-step 梯度噪声 ξ_t 是 bounded martingale-difference（无放回 shuffle 给
  E[ξ_t|F_t]=0），故 E[w_t] 服从无噪声递归，Azuma–Hoeffding 控制涨落；diag8 实测
  涨落半径 0.05-0.08 ≪ 吸引子分离 R=2.61（30-50×），noise 不能 gate crossing。
- 未动：abstract/Figure1/§intro 的 B=K·E_win 记号（等价定义，定理陈述已给换算）。

## 两个 minor 均不阻断 Gate B（A3 原判）；落笔前已处理完毕。
## T1 复审：A3 建议派 A1（调度类命题对口）——等 MGR/A1 调度，我这边 T1 文件
（PROP1_T1_FORMAL_R780.md）就绪。
