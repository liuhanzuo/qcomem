# T1: 两水平混合模型下的 majority 翻转概率与 pooled 折扣的充分性/保守度

A11 · r467 · 状态：骨架（骨架=结构正确、引理待补全证明；不得作为已证定理引用）

## 0. 动机（与 r465 实验的对应关系）

- 实验2(ii)：异质性下 iid-CP 欠覆盖、pooled 保覆盖。需要一个定理解释：**正确的证书目标是什么、为什么 pooled 碰巧保守、保守多少、能否收紧。**
- 实验3：τ² 上观测相关≈异质项、衰减项≈0。需要 T1 给出两水平分解，说明 stopper 面对的真实对象。

## 1. 模型（A-假设）

**A1（两水平交换簇）**：task g 有 N 个 rollout。潜变量
  Z_gi = √λ · U_g + √(1−λ) · E_gi,  U_g, E_gi ~ N(0,1) 独立， λ ∈ [0,1)。
观测 X_gi = 1{ Z_gi > c_g }，阈值 c_g 随 task 变。c_g ~ H（跨 task 分布，不假设参数族）。
逐 task 成功率 p_g = P(X=1|c_g) = Φ(−c_g)。异质性 = Var_H(p_g) > 0。

**A2（majority）**：Ŷ_g(k) = 1{ Σ_{i≤k} X_gi ≥ ⌈k/2⌉ }，full-N 参考 Ŷ_g = Ŷ_g(N)。

**A3（stopper）**：在前 k 个观测后输出 Ŷ_g(k)，证书要求
  P( Ŷ_g(k) ≠ Ŷ_g ) ≤ α,  概率对（H, c_g~H, E 的随机性）同时取。

## 2. 分解引理（T1-L1，观测成对相关）

对 i≠j：
  ρ_obs := Corr(X_gi, X_gj) = [ Cov_h( Φ( (m_g − c_g)/s_h )² 型双正态CDF ) + Var_g( p_g ) ] / Var(X)
即 **ρ_obs = 衰减项(latent λ, c_g 水平) + 异质项 Var(p_g)/[p̄(1−p̄)]**。

- 衰减项随 |c_g| 增大、随 p_g→0.5 衰减（probit 链接的凹性；τ² 载体上 ≈0）。
- 异质项与 λ 无关——即使 rollout 条件独立（λ=0），Var(p_g)>0 也产生正 ρ_obs。

状态：已验证（r465 synth 数值；解析式为 probit 双正态 CDF 恒等式，待写成 lemma 形式）。

## 3. 主定理骨架（T1）

**T1（混合翻转上界）**：在 A1–A3 下，
  P( Ŷ_g(k) ≠ Ŷ_g ) ≤ E_H[ τ_flip(p_g, k, N, λ) ]
其中 τ_flip 是**逐 task 条件翻转概率**（给定 p_g 与 λ 下，k-of-N 截断二项/正态桥近似的翻转概率），E_H 对 task 混合取期望。

**T1-C1（pooled 充分且保守）**：用 pooled 二阶量（p̄, ρ̂_obs）构造的证书对应
  P ≤ τ_flip(p̄_effective, k, N, λ̂_effective)，
其中 p̄_effective 吸收了 Var(p_g)（实验2(ii)的机理）。因为 τ_flip 在 p_g 关于 0.5 对称展开时是 p_g(1−p_g) 的**凹函数附近单调**，Jensen 给出
  E_H[τ_flip(p_g)] ≤ τ_flip(p̄_effective)  （当 τ_flip 在支撑上凹），
故 pooled 保守。保守量 = τ_flip(p̄_eff) − E_H[τ_flip(p_g)]，随 Var(p_g) 增大而增大。

**T1-C2（iid-CP 为何欠覆盖）**：iid 假设把 Var(p_g) 归零 → 用 τ_flip(p̄, …) 而非混合期望 → 当 Var(p_g)>0 时低估翻转概率 → 欠覆盖（实验2(ii) 实测 0.983<0.99）。

## 4. 由 T1 导出的方法（mixture-aware stopper，V5 设计）

证书 = **E_H[τ_flip(p_g, k, N)]** 的直接上界，用 CAL 集逐 task 拟合 p̂_g（empirical-Bayes 收缩：Beta 先验矩匹配 Var(p̂_g)），代入 τ_flip 逐 task 计算后取（带 Hoeffding/CLT 修正的）均值上界。停止规则：
  k* = min{ k : Û_flip(k) ≤ α }， Û_flip = E_H[τ_flip] 的 (1−δ) 上置信界。
与 V1（pooled n_eff 折扣）对比的理论预测：
  (i) 覆盖不降（同为上界，但更紧）；
  (ii) 期望停止 k 更小（去掉 Jensen 间隙）；
  (iii) 异质性越大收益越大（Var(p_g) 大时 V1 过度保守，V5 收紧）。

## 5. 待补证明（proof skeleton，四块）

- **L1** 分解恒等式：双正态 CDF 展开 + law of total covariance。（机械，半天）
- **L2** τ_flip 的显式近似：k-of-N 截断下翻转 = 尾部事件，用正态桥/精确二项给出上界；需证对交换正态簇是单调的。（核心，1-2 回合）
- **L3** 凹性/Jensen 方向：τ_flip(p) 在 p∈(0,1) 上非全局凹（尾部翻转使两端翘）。**r467 实证更新**：凹性方向依混合形状——τ²/合成异质格上 pooled 保守（T1-C1 原方向），但 OpenMathReasoning 的强 U 形混合（45% 极端 p）上 mixture-aware E_mix[flip] 反而比 uniform worst-case **紧**（0.049 vs 0.5）。结论：T1-C1 不可作全局定理，须表述为「pooled/uniform 的保守方向依混合类」的有限族命题；正贡献不依赖此方向，而依赖 V5+ 的校准 UCB（见 L4）。（风险块，方向已限定）
- **L4** 证书校准：CAL 上 Û_flip 的覆盖 ≥1−α−δ。**r467 实证发现这是承重引理**：plug-in E_mix[flip] 欠覆盖（boot p95 0.0511>0.05 at k=11），必须加不确定性强攻（bootstrap-UCB 或 Hoeffding-UCB）才有限样本有效。V5+ = E_mix[flip] + UCB-margin。boot-UCB 在 α=0.05 给出 k*=13（省 59%、violation 0.0），Hoeffding 更保守（省 53%）。L4 的证明目标：对任意问题混合， certified flip ≤ α 以 ≥1−δ 概率成立（cover the mixture expectation, not pointwise）。（实验已挂钩：v5_calib_r467_result.json）

## 6. 与实验的接口

- synth_r465 已有 (N,ρ,p,同质/异质) 全格 → 加 V5 列：实测 k、agree、翻转、与 V1 的保守差。
- 端点：agree 证书（k 越小越好，覆盖不降）+ rollout 节省。

## 7. 诚实边界

- L3 的凹性方向未证，pooled「保守」目前只在 r465 的 8 格异质合成上数值成立；T1-C1 在 L3 完成前只作解释性命题，不作定理引用。
- τ_flip 的桥近似在 N 小（≤10）时误差未量化；V5 实验必须报告近似误差 vs 精确翻转。
