# FINITE-SAMPLE SAFE MODEL RANKING UNDER SUBGROUP-MIX TURNOVER — 项目设计

## 1. 研究对象与未解决缺口
给定一组**固定候选模型** {M_i}。部署期的数据分布可在**类别/子组的混合比例**（mixture）上
相对采集期发生迁移，但**组内条件分布 P(X | Y=g) 不变**（label-prior shift / subgroup-mix
turnover）。此时模型在 mixture 权重 w (在 G 维单纯形上) 下的风险是**逐组风险的线性组合**：
```
R_i(w) = Σ_g w_g · r_{i,g},   r_{i,g} = E[loss | Y=g]
```
缺口：现有模型选择（对平均误差选优、balanced accuracy、worst-group/DRO、单 eq-mixture）
要么只用**单点估计**在采集 mixture u 上选（mixture 一移就翻），要么对 mixture 不自适应
（worst-group 无视 w 实际在哪个域）。缺一个**对每个未来 mixture w 都给出有限样本安全
排序/选择，并在不确定时回退**的框架。

## 2. 核心命题（哪个成立会让审稿人觉得非平凡）
- **P1（线性带）**：一次性 Bonferroni 同时置信区间给出逐组风险箱 [l_{i,g}, u_{i,g}]，
  使每个 w 都对应一个**线性风险带** L_i(w)=Σ w l ≤ R_i(w) ≤ U_i(w)=Σ w u，且联合覆盖
  ≥1−δ。风险排序跨 w 的**翻转集是单纯形上的半空间** {w: Σ w_g (r_{i,g}−r_{j,g})>0}。
- **P2（minimax-regret 选择）**：对选 i 的最坏情形 regret 有闭式上界
  `UB_i(w)=max_j ( U_i(w) − L_j(w) )`（对抗者把中选模型抬到箱顶、把任一竞争模型压到箱底）。
  选择 `i* = argmin_i UB_i(w)`，当 `UB_{i*}(w) ≤ τ` 时可**证书化**：中选模型在 w 处的真
  regret ≤ τ，且该选择完全只读 CAL，不需要未来数据。
- **P3（回退）**：当全部分支 UB > τ 时（w 落在带状叠加区），说明在该 mixture 处无从
  判别→诚实地报**不确定并回退**（稳健混合 / abstain），而非硬选一个可能翻车的点估计最优。

## 3. 假设与证明骨架
- A1 label-prior shift：P(X|Y=g) 在 mixture 迁移下不变（EVAL 用组内重加权实现，构造自洽）。
- A2 各候选在 CAL 上的逐组错误率 r_{i,g} 为期望的可估计量（Bernoulli），组内 iid。
- 证书：Bonferroni 联合覆盖 ≥1−δ（G·k 个 cell，δ_cell=δ/(Gk)，Clopper-Pearson 精确）。
- P2 证明：regret(i,w)=R_i(w)−min_j R_j(w) ≤ R_i(w)−R_j(w)，对取最大 j 的对抗者
  ≤ U_i(w)−L_j(w)（cell 独立箱极值）。max over j → 上界；argmin over i 得 minimax 选择。

## 4. 理论→可执行改进
MMR 规则：给定 w，用同时 CI 的线性带做 minimax-regret 选择 + 阈值证书 + 确定回退。
它比点估计最优更安全（把 CI 宽度计入），比 worst-group 更自适应（w 所在域决定带的加权），
且给出可核验的 regret 证书（审稿可复算）。**理论是主体，实验验证其在真实数据上确实
降低未来-mixture 的 regret、提高"选到 oracle 最优"命中率，并诚实给出证书/回退率。**

## 5. 最小实验 / carrier / 预注册 / 成本
- carrier：digits、Fashion-MNIST（PCA128 冻结）、MNIST（PCA128 冻结），subgroup=class。
- FIT 训固定轻量候选（LR-C1、LR-C01、LinearSVC、MLP 小网）；CAL 估逐组风险+CI；
  OUTER=封存 test，EVAL 一次只读结算（label-shift IPS 重加权估计 R_i(w)）。
- 预声明 W=12 个未来 mixture 网格（均匀 + 若干 label-skew 尖峰 + 与采集 u 的插值）。
- 端值：未来-mixture regret、rank reversal、选到 oracle 最优命中率、证书/回退率、带宽、成本。
- 成本：纯 CPU，1–5 分钟首个 readback。

## 6. 强 baseline
CAL-prior 经验最优（点估计 argmin Σw p̂）、balanced accuracy / equal-mixture、worst-group、
简单 DRO 选择、诊断 oracle。诚实边界：禁止用数据集 ID / 事后 oracle 特征；EVAL 严格 outsample。

## 7. 风险与同题转义
- 负结果（MMR 不优于点估计）：诊断是否带宽太宽（Bonferroni 过保守）→ 改进 CI（Wilson/
   pooled / 半参数）为同题修复；或回退率过高 → 调 τ、融合点估计先验。
- 全绿（所有规则近 chance）：判断是 ceiled（逐组错误率本身低，拌带无区分度）→ 换 carrier
  或改用逐组 logloss/calibration 风险而非 0-1 错误率，仍是同题。
- 排名翻转在真实数据上太少 → 用弱相关模型对构造可控翻转，仍研究翻转几何。