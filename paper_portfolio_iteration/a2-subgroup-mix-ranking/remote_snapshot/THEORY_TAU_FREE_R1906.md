# τ-Free 安全升级门（M6，r1906）— 收口冻结结论明示的 "τ-selection protocol" 缺口

同一项目 `A2_SAFE_MODEL_RANKING_SUBGROUP_MIX` 的 follow-up。冻结主稿
`subgroup_mix_ranking/paper/paper.tex`（M5-C）结论行 682 明示后续：
*tighter exact bounds and a τ-selection protocol*。本块做后半（τ-selection / 消除 τ），
不改冻结论文 bytes，只新增可复现证据与独立理论注记。

## 1. 被冻结框架遗留的操作员缺口

M2/M2.5（r1884/r1885，已冻结）的决策为
```
commit <i*(w), regret ≤ τ>   iff  UB_cf(i*,w) ≤ τ
abstain                       否则（回退到 "robust mixture / domain choice"）
```
两个遗留问题：① τ 是操作员必须事先设的旋钮，论文已诚实标注"τ 与种子是操作员/估计参数，
我们报告的是 coverage–commitment 权衡而非最优-τ 声明"；② abstain 的"回退对象"没有干净定义，
部署者无法推理"框架到底值不值得开"。结论把 "τ-selection protocol" 列为唯一敞开的同题入口之一，
但更干净的做法是**彻底消除 τ**，把安全目标从"对某绝对容忍 τ 的 regret 证书"重定位为
"相对现有无框架部署（status-quo）的**安全升级**证书"。

## 2. M6：用 status-quo F0 替换 τ

记
```
F0 = argmin_i Σ_g u_g · p̂_{i,g}      （collected-mixture 点估计最优；
                                       = 若不运行本框架、操作员实际会用的 single-point 选择器）
```
"不运行框架"的部署对象就是 F0。M6 对每个未来 mixture w 的决策为
```
decision(w) = i*(w)   iff   D(w) := Σ_g w_g · UCB_{normal}[(i*,F0)][g] ≤ 0
            = F0       否则（诚实回退 = 继续跑 status-quo）
```
其中 `UCB[(i,j)][g]` 是冻结 M2.5 流水线同一个 `d_{ijg}=mean_x[err_i-err_j|g]` 的
paired-difference 单侧 normal UCB，Bonferroni 分裂 `dcell=δ/(M(M-1)G)`（joint 覆盖 ≥1−δ，
对全有序对×组同时成立，因而对 CAL 选出的 i* 与 F0 条件化安全）：
```
D(w) ≤ 0  ⇒  R_{i*}(w) − R_{F0}(w) ≤ Σ_g w_g (D(w)) ≤ 0   （joint event 上）
```
即：committed 选择在 **这个 w** 上被证成"不比 status-quo F0 差"，**全程无 τ**。

### 诚实标注的语义边界
- 这是**相对**证书（no-worse-than-F0），不是"regret 绝对 ≤τ"的绝对证书；两者互补。
  若操作员的真实需求是"必须保证 regret ≤ τ"，仍用冻结 M2.5；若需求是"框架不能比
  现在更糟"，M6 直接用这不是旋钮。
- 单侧 normal UCB 是渐近（CLT）带，与冻结前沿线的 Normal 一行同级；若需保证，替换成
  Hoeffding/MPB 同公式即可（决策规则当然随之变保守）。本块沿用冻结主表的 normal 口径。
- 没有免费的绝对后悔声明：`or_max_committed`（对 oracle-best 的 regret）如实披露，不作
  "≤τ"包装。

## 3. 忠实复现断言

M6 脚本在判定 M6 数字可信前，先对冻结 `SUBGMIX_M25_PAIRED_R1885_5SEED.json` 逐行断言
`chosen / true_regret / UB_paired / committed` 四个共享字段 **逐位一致**（`assert not mism`）。
因为 M6 复用同一 split/同一点估计选择器/同一条 paired-UCB 矩阵（只是换把 max_j 换成与 F0
配对的单跳），复现一致是 M6 事故侧可审计性的前提。前台 EXIT=0，base 字段零漂移。

## 4. 真实证据（5-seed 全载体，front，EXIT=0，`results/SUBGMIX_M6_UPGRADEGATE_R1906.json`）

- upgrade_rate（框架在多少 w 上切换脱离 F0）= **0.663**（mnist 1.0 / digits 0.773 /
  fashion 0.560 / news 0.456）。
- **REG_sq = R_decision − R_F0（对 status-quo 的 regret）**：committed（upgraded）与全体
  **max = 0.0、mean < 0**（全体 −0.0015；upgraded −0.0022），`sq_no_worse_cov = 1.0`
  —— 无一行让框架比"不运行框架"更差。τ-free 且 sound（OUTER 结算）。
- **机制**（论文主线：单点选择在 turnover 处失效）：升级收益集中在 skew
  w 上（upgrade 后 mean_gain +0.0035、最大 +0.130），uniform/collected/interp 上增益≈0
  （该处 F0 已近最优，升级只是 validate 不伤害）。
- **门的价值（counterfactual）**：M6 在 118 个 w 上 abstain 保 F0；其中 naive 恒切到
  i* 会在 **23 个（19%）真变差**（REG_sq 正至 +0.028）。跨全部 350 个 w：M6 恒
  REG_sq ≤ 0（max 0.0），而 naive always-switch 的 REG_sq max +0.028。即"自适应点估计选择
  器"单独不可靠，M6 门把它的有害翻转全部挡住，又保留其在可由数据证成的 w 上的收益——
  == "point-estimate selection + certificate gating"（论文 M2/M2.5 的同一原则）在相对
  status-quo 意义上的直接兑现。
- 绝对后悔透明度：全体 `or_mean=0.0079`、or_max(committed)=0.0546、or_max(abstain)=0.1496；
  不作"绝对 ≤τ"声明。

## 5. 与冻结稿件的关系 → 后续写作入口

- 不改冻结 bytes。M6 补上结论明示的 τ-selection 这一半；在一个可能的新增 § or 附录
  （若 MGR 签发迁稿后可加，独立于已冻结主稿）中报告，或作为独立 follow-up 注记。
- 理论身份与既有 Thm 1（MCB/paired-difference + Bonferroni joint event）同一来源，M6 只是
  把证书**被控对象**从 oracle-best 换成 status-quo F0——这实质是"certificate target 选的
  是绝对最优还是相对现部署"的合法分离点，与 A6/A12 的"证书窄口径到被释放动作"一脉相承。
- 下一页：① 若审稿要"绝对 regret ≤ τ"仍给冻结 M2.5；② M6 与 M2.5 是同一合成数据/同 split
  的互补端点，不是二次结果。截止本块为止仍是纯 CPU。

## 6. 诚实边界 / 下一步
- single-side normal（渐近）。可加 Hoeffding/MPB exact 变体看 upgrade_rate 掉多少（不阻断）。
- F0=collected-mixture single-point best 只是"无框架"部署的一种合理选择；若操作员本就要跑
  robust/DRO 基线，F0 定义相应换——M6 结构不变（相对"实际会跑的"对象）。
- snowflake：升级收益 >0 的绝对量级依赖 carrier 异质性（news/digits 可证成、fashion 小），
  正则其"至少不差"：无框架相对安全，可证成收益是何时值得升级。