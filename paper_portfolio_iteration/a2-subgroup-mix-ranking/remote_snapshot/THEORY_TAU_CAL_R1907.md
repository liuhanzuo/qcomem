# 有限 τ 菜单的 CAL-only 选择协议（M7，r1907）— 同时有效性 + 选择代价

同一项目 `A2_SAFE_MODEL_RANKING_SUBGROUP_MIX`。冻结主稿（M5-C）结论明示
*tighter exact bounds and a τ-selection protocol*；M6（r1906）用 status-quo F0
消除了 τ。本块做第三选择：**保留绝对-τ 语义，但让操作员从有限菜单里用 CAL-only
数据选 τ**，形式化"选择后的同时有效性"与"选择代价"。纯 CPU / 前台 / 零 GPU。

## 1. 冻结证书的结构（τ-agnostic band）

冻结 M2.5（r1885，已冻结 `SUBGMIX_M25_PAIRED_R1885_5SEED.json`）对每个 carrier×seed
构建 paired-difference 单侧 normal UCB，Bonferroni 分裂 `dcell = δ/(M(M-1)G)`：
```
UCB[(i,j)][g] = mean_{x∈CAL, y=g} [err_i(x)-err_j(x)] + z·σ_g/√n_g ,  z=Φ⁻¹(1−dcell)
D(i,w) = max_{j≠i} Σ_g w_g UCB[(i,j)][g]
committed_τ(w) = 1{ i*(w)=argmin_k Σ_g w_g p̂_k,g  且  D(i*,w) ≤ τ }
```
Prop P1 给出联合覆盖：对所有有序模型对×所有组，(对称/单侧变换后)
`D(i*,w) 对所有 w∈网格、所有模型对 同时成立` 在概率 ≥1−δ 事件上。**τ 只出现在
`D(i*,w) ≤ τ` 的比较中，不进 band 的构造。** 因而 band 的覆盖概率对 τ 完全不变。

## 2. 预注册：有限 τ 菜单与总 δ

- 菜单 `T = {0.01, 0.02, 0.03, 0.04, 0.05}`，`L=5`；冻结默认 τ=0.04 ∈ T（作为固定-τ 基线）。
- 总错误预算 δ=0.10，与冻结一致，**不因 L=5 增加 band 多重性**（见 §3）。
- 选择器只用 CAL 快（新 block 的 CAL 组误差 `p̂_{k,g}` 与 band 同一批，见 §4），
  绝不读 OUTER/test 来决定 τ̂。

## 3. 定理 Pro【τ 选择后的同时有效性】

设冻结 band 联合覆盖 ≥1−δ（对所有有序模型对×组，Prop P1）。设 `τ̂ = g(D_cal)`
为任何**只依赖 CAL 块**（fit/cal 组误差，不含 outer/test）的选择规则（如原始目标函数
选出的阈值、用 CAL 点估计算出的 committed-rate floor 反解，甚至任意确定性常数）。
则：

1. **有效性不因选择而坏**：对每个 w，在联合 band 事件上，
   `committed_{τ̂}(w)=1 ⇒ true_regret(w) ≤ τ̂`。且 `P(联合事件) ≥ 1−δ`
   对任意 g 成立——因为 band 不含 τ，选择 g 不改变覆盖概率。
2. **band 上零额外多重性代价**：候选数 L 不进入 band 的 dcell。选择 τ̂ 只改变
   *哪些点被 commit*，不改变每个被 commit 点的覆盖。对 `T={0.01..0.05}`，
   每点覆盖仍是第 P1 层（≥1−δ），不是 ≥1−δ/L。

推论【无校正诊断的反面】若选择器读 test/OUTER 来挑 τ（数据窥探），则"我 commit
这些点且 regret≤τ̂"仍是逐点 band-有效（因为 band 不含 τ），但**报告的聚合
性能（committed rate、均值/最大 regret）变得乐观**：同一份 test 既选了 τ 又报告了
取决于 τ 的统计量，选择与评估共用数据产生选择偏差。这正是"无校正诊断"要展示的量。

**形式化（选择代价）**：记方法在 τ 下的真实测试性能为 `Perf(τ|test)`（如
`committed_rate` 或 `mean committed regret`）。oracle-τ = argmax_{T} Perf(τ|test)
（事后最好，看全 test）。CAL-only τ̂ 的选择代价分解为
```
SelCost = Perf(τ*_oracle) − Perf(τ̂)   = (i) 代理失配 [CAL 目标 ≠ test 真实]
                                       + (ii) 网格离散 [T 有限]
                                       + (iii) 抽样方差 [τ̂ 在 CAL 上的随机性]
```
(i)(ii) 是诚实代价下限，(iii) 用 paired 区间报告 Δ 是否统计上非零。由于 band
同时有效，**无论 SelCost 多大，被 commit 点的 regret≤τ̂ 保证不坏**——代价只在
性能层，不在证书层。

## 4. 全新不相交 mixture/seed block（诚实评估）

- 冻结证据用 seed {0,1,2,3,4}（M2.5/M6 已报）。本块用**新 seed {5,6,7,8}**：
  与冻结完全不相交，重新走同一冻结 pipeline（fit/cal/outer 切分，模型照冻结
  model_pool），prediction 结构（mist/UCB/oracle/w-grid）一致。此 block 从未被报过，
  是"全新不相交 block"。
- 每 carrier 每 seed：CAL 同时用于 band 和 τ̂ 选择。这是合法的：band 同时有效与
  选择无关（定理 3.1），选择读自己的 CAL 不构成 test 窥探。
- OUTER/test 只用于**结算**（oracle regret、committed rate、coverage），与冻结口径一致。

## 5. 臂与判据（同预算、可配对）

对每 carrier×seed×w（w-grid），结算层面的真实 regret 来自 OUTER：
```
true_regret(w) = R_{decision}(w) − min_j R_j(w)   （决定 = committed ? i* : abstain）
```
五臂（同 θ/同 δ/同 grid，只有 τ 来源不同）：
1. **固定-τ**（对每个 τ∈T 各跑一次；冻结默认 0.04 为主基线）：`committed  iff D(i*,w)≤τ`。
2. **CAL-only 选 τ̂**（本协议）：`τ̂ = min{τ∈T : CR_cal(τ) ≥ p0}`，p0=0.50，
  空则取 T 中最小 τ（lazy/safe 回退）。
3. **同时有效证书（=臂 2 的覆盖）**：report `true_regret(w) ≤ τ̂` 的被 commit 比例
  （OUTER 结算）。预期 ≈1.0（band 同时有效，定理 3.1 的实证兑现）。
4. **无校正诊断（naive）**：直接在 OUTER/test 上挑 `τ*_naive = argmax_T CR_test(τ)`，
  再在同一 test 报聚合 → 展示数据窥探把 committed rate 与表现虚高。
5. **oracle-τ**：`τ*_oracle = argmin_T [mean committed test-regret，惩罚低 CR]`
  （看全 test 的事后最好）→ 选代上界。

**同预算**：所有臂共享同一 CAL/OUTER 数据与同一 band；只有 τ 的出处不同，
不额外消耗标签/算力（标签成本 = 选 τ 块+CAL+OUTER 读数均已发生，与固定-τ 一致引
用冻结口径）。

## 6. 报告

- `committed rate`、`mean/max true regret(committed)`、`coverage(true_regret≤τ̂)`。
- **paired 区间**：per-(carrier,new-seed,mixture) 对 `Δ = Perf(τ̂) − Perf(0.04)`
  的 t·SEM paired 置信区间（诚实报告，单元=seed×mixture，非 cell 独立）。
- **全部弱域**：per-carrier 拆解（news/digits 预算墙、fashion 小异质性、低覆盖 cell
  如实留）；任何 committed=0 或 coverage<1 的域不隐藏。
- **SelCost** = Perf(oracle-τ) − Perf(τ̂)，分解注释 (i)(ii)(iii)。
- 承诺：无论收益大小，解释/条件化/融合并写回 RESULT_MATRIX/REPRO/主稿注记。

## 7. 诚实边界

- band 为单侧 normal（渐近）；需 exact 可换 Hoeffding/MPB，规则随之保守（冻结主表
  口径为 normal）。
- `τ̂` 的 floor 规则 p0 是刚预注册的单一规则，不是扫 p0。
- cal-only 选 τ 的 CR_cal 用 CAL 点估计（diag 用），只算选择，不泛化为"CR 预测"。
- 与新 seed 相关的 carrier 差异属 seed 噪声，paired 区间吸收。