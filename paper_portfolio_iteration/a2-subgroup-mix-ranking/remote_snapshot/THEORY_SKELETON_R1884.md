# FINITE-SAMPLE SAFE MODEL RANKING UNDER SUBGROUP-MIX TURNOVER — 理论骨架

项目：A2_SAFE_MODEL_RANKING_SUBGROUP_MIX。状态：证明中；已由 r1884 真实读回验证数值自洽。
与 r1881 M1 负结果的对照见 §5。

## 设定（固定定义）
- 候选模型 I={1..k}，ROUND以**组(ground-truth class g∈{1..G})**为单位。
- CAL 数据：每组 g 有 n_g 个样本（该组真错误率 Bernoulli 估计）。
- 对每个 (i,g)：点估计 p̂_{i,g}=#{err|g}/n_g，CP 精确 CI [l_{i,g},u_{i,g}]，每格置信 δ_cell=δ/(Gk)。
- Bonferroni：P( 所有 (i,g) 同时满足 r_{i,g}∈[l,u] ) ≥ 1−δ。
- mixture w∈simplex：R_i(w)=Σ_g w_g r_{i,g}，线性带 L_i(w)=Σ w l, U_i(w)=Σ w u。

## Lemma 1（线性带联合覆盖）
若 r_{i,g}∈[l_{i,g},u_{i,g}] 对所有 (i,g) 同时成立（事件 E，P(E)≥1−δ），则对任意 w∈simplex、
任意 i：L_i(w)≤R_i(w)≤U_i(w)。因 L_i,U_i 是 w 的线性组合，w 确定性，带覆盖随单格覆盖转移。
证：E∩{w=const} 下逐项 w_g∈[0,1],Σw=1，不等式按单调性组合立得。∎

## Proposition 2（点估计选择的 sound regret 上界）
记 i*∈argmin_i pt_i(w), pt_i(w)=Σ w p̂。定义
`UB_{i*}(w) = U_{i*}(w) − min_j L_j(w)`。
则在事件 E 下：`regret(i*,w) = R_{i*}(w) − min_j R_j(w) ≤ U_{i*}(w) − min_j L_j(w) = UB_{i*}(w)`。
证：R_{i*}(w)≤U_{i*}(w)（Lemma 1），且 min_j R_j(w) ≥ min_j L_j(w)（因每 R_j≥L_j）。相减即得上界。∎
注意：此界对**任一** i* 成立，不只 min-pt；它是"所选模型回报最坏情形"的一致上界。

## Definition（cert-gated 选择 / P3 回退）
在部署 mixture w 处：
- 若 `UB_{i*}(w) ≤ τ`：**committ**——输出 i* 并携带证书"真 regret≤τ"（在 E 上成立，P(E)≥1−δ）。
- 否则：**abstain**——诚实报告 w 处无从判别；资方/上层用 fallback（稳健混合或多模型投票）而非硬选。

## Theorem 1（证书健全与覆盖-宽裕权衡）
在 E（P(E)≥1−δ）下，所有 committed 决策满足真 regret≤τ；即证书**健全**。
随 τ↑，committed 集增大（覆盖放宽），上限承诺随之放宽；τ↓ 覆盖更保守。权衡由
`committed_rate(τ)=#{w:UB≤τ}/#{w}` 刻画（r1884 已实测，见 RESULT_MATRIX）。

## 可操作端点
- 覆盖：certified 决策的覆盖保证 ≥1−δ（Bonferroni 联合）。
- 宽裕：τ 是用户设置的最大 regret 容忍，容器由 CP CI 宽度决定。
- 回退：不能证书的 mixture 恰是点估计最可能翻车的 turnover-skew 区（r1884 实测 abstain
  hardpick regret 均值 0.0043、最大 0.0546，远高于 committed 的 0.0001/0.0019）。

## 待证明/待补（诚实标注）
- TBD-1 **已解决**（r1898）：配对证书自动为 simplex-同时。UCB[(i,j)][g] 独立于 w（只在有序对×组上
  Bonferroni δ/(M(M−1)G)），同一事件 E 同时覆盖整个连续单纯形上所有 w 与数据依赖选择 i*(w)；
  无需对 w 网格再 Bonferroni。r1898 以每 cell 1000 离网格 Dirichlet 扫描核验：committed 覆盖率仍
  1.0。**更正旧"per-w 逐点 / 网格联合待做"文句：该自我降级对配对证书过于保守。**（绝对带 M1/M2
  变体另有其逐 w 性，见 M1 段落。）
- TBD-2 半参数/更紧 CI（Wilson、pooled、logloss-risk）替换 CP 是否维持覆盖并提高 committed_rate
  的数值；作为同题修复分支待做。
- TBD-3 定理只给"健全性"（覆盖），不承诺"最优回退策略"；最优 τ 是设计参数（报告权衡），非定理结论。

## §6 M2.5（r1885）：配对差（MCB 风格）证书 — 收窄宽带的同题修复

r1884 M2 的 `UB=U_{i*}-min_j L_j` 用两个**绝对** CP 带相减，宽（digits/news 中位 UB≈0.17），
而真 regret 很小（r1884 中位 0.0000、max 0.0546）→ committed_rate 被绝对带保守性门住，非被
真实模型差异门住（对照 r1884 RESULT_MATRIX）。同题修复 M2.5：证书目标是 `regret(i*,w)=max_j
(R_{i*}-R_j)`，而每个差 `R_{i*}(w)-R_j(w)=Σ_g w_g (r_{i*,g}-r_{j,g})` 是 **CAL 内配对差**（同点、
同组）→ 共享误差抵消，比减两个独立绝对带紧得多。

**Proposition 3（配对差自举界）** 对每个有序对 (i,j) 与组 g，令
d_{ijg}=mean_{x∈CAL,y=g}[1{err_i(x)}-1{err_j(x)}]，对 r_{i,g}-r_{j,g} 无偏。误差单元 X=(d+1)/2∈[0,1]
（自举差），其一侧尾可被以下之一上界：
  (a) CLT/Normal：d ≥ mean + z_{α}·sd/√n    （渐近，最紧）
  (b) Hoeffding：X ≥ mX + √(2 ln(2/α)/n)    （精确·无分布·最宽）
  (c) Maurer–Pontil 经验-Bernstein：X ≥ mX + √(2 vX ln(2/α)/n) + 7 ln(2/α)/(3(n-1))（精确·利用方差·居间）
  α=dcell=δ/(M(M-1)·G)（Bonferroni 跨所有有序对×组）。

**Theorem 2（M2.5 证书健全）** 在联合事件 P(E)≥1-δ 下，对所选 i*：
`UBcf(i*,w)=max_{j≠i*} Σ_g w_g UCB_{i*j,g}`；若 `UBcf≤τ` 则 regret(i*,w)≤τ。健全性同 Prop2，
但带更窄。选择仍由点估计 argmin pt 驱动（不用带当选择器，规避 M1）；带只做门/证书。

**r1885 真实读回（3 seed ×4 carrier，210 行，前台 EXIT=0；全精确/渐近变体 cert_cov=1.0）**：
| 证书 | committed_rate | cert_cov | 类型 |
|---|---|---|---|
| M2 absolute 带（r1884） | 0.269 | 1.0 | 精确·CP |
| M2.5 Hoeffding | 0.09 | 1.0 | 精确·无分布 |
| M2.5 Maurer–Pontil | 0.257 | 1.0 | 精确·利用方差 |
| M2.5 Normal(CLT) | 0.495 | 1.0 | 渐近|

解读：①**精确** M2.5(MPB) committed_rate=0.257 **略低**于 M2 0.269（无配对不劣检验即不写"不降/非劣"；
配对方差项有时比减两绝对带宽更宽）。②**渐近** Normal 变体 0.495 仅诊断（有限样本不健全），不能作
headline。③证书价格前沿=变体→健全性→覆盖的权衡，写为正文机制分析（不是 cherry-pick）。
④每载体：fashion/mnist 全 cert（低组内错误率→窄带）；digits/news（组错误率高→宽带→诚实低覆盖）。
⑤M1 数字纠偏（盘面 JSON 逐行核对）：mixture-ball 352 顶点实际 **165/352 diverge**（非 88/352），
diverged 子集 regret diff **−0.0021**（mrr 略优，93 better/62 worse，无一致方向），overall +0.0019；
diag 实际 15/880（非 4/880）；分歧行 11/15 tied、4/15 M1 略劣（diff +0.0024，r1901 精确核对）。
M1 是"带做门不做选择器"的机制警，非"恒反选"数据结论。

## 诚实边界 / 未做（M2.5）
- 精确覆盖验证在**独立 OUTER** 上行真 regret（只读）；M2.5 证书健全性 claim 是"在 CAL 联合事件 E、
  P(E)≥1-δ 上真 regret≤τ"。因 UCB[(i,j)][g] 独立于 w，E 同时覆盖整个连续单纯形上所有 w 与
  数据依赖选择 i*(w)（simplex-同时，r1898 离网格 Dirichlet 扫描核验 committed 覆盖仍 1.0；
  TBD-1 已由构造+实测关闭）。
- Normal 变体为**渐近**健全（CLT），不是有限样本；正文须与精确变体(MPB/Hoeffding)并列、不冒认。
- 3 seed（非 r1884 的 5 seed）为可前台完成而裁剪；完整 5 seed×4 carrier 复验见 TBD-4 同题待办。
- 最优 τ 仍是设计参数（报告权衡曲线），非定理结论。

## §5 与 M1 负结果的同题关系
r1881 naive `argmin_i U_i(w)` 已证反效果（minimax-regret 选择撞上 max_j(U_i−L_j)=U_i−min_j L_j，
退化为 argmin U_i，异质宽度下被悲观端驱动而 regret 增大）。本文不把带当选择器，只当**证书门**，
把选择权留在点估计——保证至少不差于点估计，同时给出诚实证书与回退率。属同题修复。